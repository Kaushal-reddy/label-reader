# Two images from one file:
#   docker build --target app      -t label-reader-app .       Streamlit asset register (port 8501)
#   docker build --target mock-erp -t label-reader-mock-erp .  the mock ERP (port 8000)
# docker-compose.yml runs both together. Neither image contains real photos or
# serials (see .dockerignore); the app ships SYNTHETIC demo photos in /app/demo.

ARG PYTHON=3.14-slim-trixie

# ---------------------------------------------------------------- app
FROM python:${PYTHON} AS app

# Tesseract and ZBar are the native programs pip can't install; libgl1 and
# libglib2.0 are what OpenCV's wheel needs on a slim image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr libzbar0 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt requirements-erp.txt ./
RUN pip install -r requirements.txt -r requirements-erp.txt

COPY app.py pyproject.toml README.md truth.example.csv ./
COPY label_reader/ label_reader/
COPY scripts/ scripts/
# Fake label photos to try the app with (and a truth.csv to seed the ERP from).
RUN python -m scripts.make_synthetic_dataset demo

# The code stays root-owned (read-only for the app); the user may only create
# output/ next to it (pipeline runs) and write to /data (the register).
RUN useradd --create-home --uid 1000 reader && mkdir /data && chown reader /data /app
USER reader
ENV ASSET_DATA_DIR=/data ERP_URL=http://erp:8000
VOLUME /data
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

# ---------------------------------------------------------------- mock ERP
FROM python:${PYTHON} AS mock-erp

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt requirements-erp.txt ./
# Only the ERP's packages, plus pandas (and its pinned dependencies) for `seed`.
RUN pip install -r requirements-erp.txt \
      $(grep -E '^(pandas|numpy|python-dateutil|six|tzdata)==' requirements.txt)

COPY label_reader/__init__.py label_reader/
COPY label_reader/erp/ label_reader/erp/
# The demo job's manifest: the synthetic photos' truth.csv, made in the app stage.
COPY --from=app /app/demo/truth.csv demo/truth.csv

RUN useradd --create-home --uid 1000 erp && mkdir /data && chown erp /data
USER erp
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
ENTRYPOINT ["python", "-m", "label_reader.erp.mock_erp", "--db", "/data/erp.sqlite"]
CMD ["serve", "--host", "0.0.0.0"]
