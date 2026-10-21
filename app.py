"""A small asset register ("mini-ERP") on top of the label reader: the screens.

Run with:  streamlit run app.py   (from the repo folder, with the venv active)
It opens in the browser at http://localhost:8501. For the ERP parts, also run
the mock ERP in a second terminal:  python mock_erp.py serve
(after seeding a job once:  python mock_erp.py seed --job JOB-0001 --from truth.csv)

Streamlit re-runs this whole file from top to bottom every time someone clicks
or uploads something, and draws whatever the code says to draw. So there are no
callbacks: the page is just "read the inputs, show the outputs", top to bottom.

Screens (picked in the sidebar):
  Scan     upload a photo -> barcode + OCR + validation (Stages 3-6) -> save, and post to the ERP
  Review   items the reader wasn't sure about: a person corrects / confirms them
  Assets   the register: search, filter, assign items to people, export to CSV
  ERP      the selected job: what's received, what's waiting to send, what needs a decision

Storage and the ERP link are in register.py (no Streamlit there).
"""

import os
import time

import cv2
import numpy as np
import streamlit as st

import register as reg
from erp_client import ErpError
from ocr_reader import read_ocr
from barcode_reader import read_barcode
from validate import barcode_candidates, decide, looks_like_serial, ocr_candidates
from baseline import valid_ean13


# ---------- the label reader ----------

def scan_photo(img: np.ndarray) -> dict:
    """Run the same steps as validate.py on one photo, and time them."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    values, _, conflicts, agreeing = read_barcode(gray)
    t_barcode = time.perf_counter() - t0
    t0 = time.perf_counter()
    text = read_ocr(img)
    t_ocr = time.perf_counter() - t0

    bc = barcode_candidates(values)
    ocr_lab, ocr_unlab = ocr_candidates(text)
    status, serial, source, reason = decide(bc, ocr_lab, ocr_unlab, conflicts, agreeing)
    return {"status": status, "serial": serial, "source": source, "reason": reason,
            "barcode_values": values, "conflicts": conflicts,
            "ocr_labelled": ocr_lab, "ocr_unlabelled": ocr_unlab, "ocr_text": text,
            "t_barcode": t_barcode, "t_ocr": t_ocr}


def format_problem(value: str) -> str | None:
    """Why a typed value doesn't look like an ID, or None if it does.

    Uses the same rules as the pipeline (validate.looks_like_serial,
    baseline.valid_ean13), so a person can't confirm something the reader
    itself would never accept as an ID. It's a warning, not a block: a real
    serial from another manufacturer might not fit the generic pattern.
    """
    if not value:
        return "The value is empty."
    if value.isdigit():
        if len(value) == 13 and not valid_ean13(value):
            return "13 digits, but the EAN-13 check digit is wrong - probably a typo."
        if len(value) != 13:
            return f"{len(value)} digits: a product barcode number has 13."
        return None
    if not looks_like_serial(value):
        return "Doesn't look like a serial (10-20 letters/digits, at least 1 letter and 2 digits)."
    return None


# ---------- small display helpers ----------

def show_photo(path: str) -> None:
    """Show a saved photo, or say why it can't be shown.

    A missing or damaged file must not crash the whole screen: the other
    items in the queue still need to be reviewed.
    """
    if not os.path.exists(path):
        st.warning(f"Photo missing: {path}")
        return
    try:
        st.image(path, width="stretch")
    except Exception as e:   # Streamlit raises PIL's error for a file that isn't an image
        st.warning(f"Photo can't be shown ({type(e).__name__}): {path}")


def show_erp_result(status: str, detail: str) -> None:
    """One coloured line for what happened when an item was sent to the ERP."""
    label = reg.ERP_LABELS.get(status, status)
    if status == reg.ERP_SENT:
        st.success(f"ERP: {label}. {detail}")
    elif status in (reg.ERP_PENDING, reg.ERP_OFFLINE):
        st.info(f"ERP: {label}. Saved in the register; it can be sent from the ERP screen later.")
    else:
        st.warning(f"ERP: {label}. {detail}")


def show_match_preview(serial: str, job_id: str | None) -> None:
    """Where this value would go on the job's manifest, before anything is posted."""
    if not job_id or not serial:
        return
    try:
        m = reg.match_preview(serial, job_id)
    except ErpError as e:
        st.caption(f"Manifest check unavailable: {e}")
        return
    if m.category == "exact":
        st.caption(f"Manifest: matches line {m.line['line_id']} ({m.line['description']}) on {job_id}.")
    else:
        st.caption(f"Manifest: {reg.ERP_LABELS.get(m.category, m.category)}. {m.reason}")


# ---------- screens ----------

def scan_screen(job_id: str | None) -> None:
    st.header("Scan a label")
    upload = st.file_uploader("Photo of a serial label or product barcode",
                              type=["jpg", "jpeg", "png"])
    if upload is None:
        st.info("Upload a photo to read it.")
        return

    photo_bytes = upload.getvalue()
    # Streamlit gives us the file's bytes; OpenCV decodes them straight from memory.
    img = cv2.imdecode(np.frombuffer(photo_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("That file couldn't be read as an image.")
        return

    left, right = st.columns([1, 1])
    left.image(img, channels="BGR", width="stretch")

    # Streamlit re-runs the file on every click (e.g. the Save button below), and
    # reading takes seconds. st.session_state survives re-runs, so the result is
    # kept per uploaded file and the photo is only read once.
    key = f"result_{upload.file_id}"
    if key not in st.session_state:
        with st.spinner("Reading barcode and text..."):
            st.session_state[key] = scan_photo(img)
    r = st.session_state[key]

    with right:
        if r["status"] == "accepted":
            st.success(f"Accepted: **{r['serial']}**")
        else:
            st.warning(f"Needs review. Best guess: **{r['serial'] or 'nothing read'}**")
        st.write(r["reason"])
        st.caption(f"Barcode {r['t_barcode']:.1f} s, OCR {r['t_ocr']:.1f} s")
        show_match_preview(r["serial"], job_id)

        earlier = reg.already_saved(r["serial"]) if r["serial"] else []
        saved_key = f"saved_{upload.file_id}"
        if earlier and saved_key not in st.session_state:
            st.info(f"Already in the register {len(earlier)} time(s), first on "
                    f"{earlier[0]['scanned_at']} (status: {earlier[0]['status']}).")

        # Accepted items go straight on to the ERP; uncertain ones wait for the Review screen.
        will_post = job_id is not None and r["status"] == "accepted"
        if saved_key in st.session_state:
            item_id, erp_result = st.session_state[saved_key]
            st.success(f"Saved as item #{item_id}.")
            if erp_result:
                show_erp_result(*erp_result[:2])
            elif r["status"] != "accepted":
                st.info("It's in the Review queue; it goes to the ERP once a person confirms it.")
        elif st.button("Save and send to ERP" if will_post else "Save to register",
                       type="primary", disabled=not r["serial"]):
            item_id = reg.save_item(photo_bytes, upload.name, r)
            erp_result = None
            if will_post:
                with st.spinner("Sending to the ERP..."):
                    erp_result = reg.post_item(item_id, job_id)
            st.session_state[saved_key] = (item_id, erp_result)
            st.rerun()   # redraw now, so the "Saved" message replaces the button

    with st.expander("What the reader saw"):
        st.write("**Barcode values:**", r["barcode_values"] or "none")
        if r["conflicts"]:
            st.write("**Rejected barcode reads:**", r["conflicts"])
        st.write("**OCR candidates after an SN key:**", r["ocr_labelled"] or "none")
        st.write("**Other OCR candidates:**", r["ocr_unlabelled"] or "none")
        st.text_area("Raw OCR text", r["ocr_text"], height=200)


def review_screen(job_id: str | None) -> None:
    st.header("Review queue")
    items = reg.review_items()
    if not items:
        st.success("Nothing to review.")
        return
    st.write(f"{len(items)} item(s) the reader wasn't sure about. Check each photo, "
             "correct the value if needed, then confirm."
             + (f" Confirmed items are sent to {job_id}." if job_id else ""))

    for item in items:
        value_key = f"value_{item['id']}"
        # The value lives in session_state (not value=...), so the "Use this"
        # buttons below can fill it in from a manifest line.
        if value_key not in st.session_state:
            st.session_state[value_key] = item["serial"]

        # border=True draws a box around each item so they're easy to tell apart.
        with st.container(border=True):
            left, right = st.columns([1, 1])
            with left:
                show_photo(item["photo_path"])

            with right:
                st.subheader(f"Item #{item['id']}")
                st.caption(f"Scanned {item['scanned_at']}")
                st.write(f"**Why it needs review:** {item['reason']}")

                # A form groups the inputs: typing doesn't re-run the page,
                # only pressing a button does. The key keeps each item's form separate.
                with st.form(key=f"review_{item['id']}"):
                    # Uppercased and stripped: labels print serials in capitals, and
                    # stray spaces from copy-paste would make the same serial look different.
                    value = st.text_input("Serial / product number", key=value_key).strip().upper()
                    notes = st.text_input("Notes (optional)", value=item["notes"] or "")
                    problem = format_problem(value)
                    if problem:
                        st.warning(problem)
                    confirm_col, delete_col = st.columns(2)
                    confirmed = confirm_col.form_submit_button(
                        "Confirm and send" if job_id else "Confirm", type="primary", disabled=not value)
                    deleted = delete_col.form_submit_button("Delete scan")

                show_match_preview(value, job_id)
                use_manifest_buttons(item["id"], value, job_id, value_key)

                if confirmed:
                    reg.confirm_item(item["id"], value, notes)
                    message = f"Item #{item['id']} confirmed as {value}"
                    if job_id:
                        status, _, _ = reg.post_item(item["id"], job_id)
                        message += f"; ERP: {reg.ERP_LABELS.get(status, status)}"
                    del st.session_state[value_key]
                    st.toast(message)
                    st.rerun()   # redraw: the confirmed item leaves the queue
                if deleted:
                    # Deleting can't be undone, so ask once more.
                    st.session_state[f"really_delete_{item['id']}"] = True
                if st.session_state.get(f"really_delete_{item['id']}"):
                    st.error("Delete this scan and its photo? This can't be undone.")
                    yes, no = st.columns(2)
                    if yes.button("Yes, delete", key=f"yes_{item['id']}"):
                        reg.delete_item(item["id"])
                        del st.session_state[f"really_delete_{item['id']}"]
                        st.session_state.pop(value_key, None)
                        st.rerun()
                    if no.button("Cancel", key=f"no_{item['id']}"):
                        del st.session_state[f"really_delete_{item['id']}"]
                        st.rerun()


def use_manifest_buttons(item_id: int, value: str, job_id: str | None, value_key: str) -> None:
    """For a near match, one button per manifest line it could be: fills in that line's identifier.

    The person still has to press Confirm, so nothing is changed without a human
    looking at the photo and the line side by side.
    """
    if not job_id or not value:
        return
    try:
        m = reg.match_preview(value, job_id)
    except ErpError:
        return
    if m.category != "near_match":
        return
    for line in m.candidates:
        # on_click runs BEFORE the next re-run draws the page. Setting the text box's
        # value after it's been drawn is an error in Streamlit, so it has to happen here.
        st.button(f"Use line {line['line_id']}: {line['identifier']} ({line['description']})",
                  key=f"use_{item_id}_{line['line_id']}",
                  on_click=st.session_state.__setitem__, args=(value_key, line["identifier"]))


def assets_screen() -> None:
    st.header("Assets")
    df = reg.all_items()
    if df.empty:
        st.info("The register is empty. Scan a label to add the first item.")
        return

    # Headline numbers for the whole register (before any filter).
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Items", len(df))
    c2.metric("Devices", int((df.item_type == "device").sum()))
    c3.metric("Products", int((df.item_type == "product").sum()))
    c4.metric("Waiting for review", int((df.status == "needs_review").sum()))
    c5.metric("Received on ERP", int((df.erp_status == reg.ERP_SENT).sum()))

    f1, f2, f3 = st.columns([2, 1, 1])
    query = f1.text_input("Search serial, person or notes").strip().lower()
    statuses = f2.multiselect("Status", ["accepted", "confirmed", "needs_review"],
                              default=["accepted", "confirmed"])
    types = f3.multiselect("Type", ["device", "product"], default=["device", "product"])

    shown = df[df.status.isin(statuses) & df.item_type.isin(types)]
    if query:
        # regex=False: treat what was typed as plain text, so e.g. "." or "(" can't break the search.
        hit = (shown.serial.str.lower().str.contains(query, regex=False)
               | shown.assigned_to.str.lower().str.contains(query, regex=False)
               | shown.notes.str.lower().str.contains(query, regex=False))
        shown = shown[hit]
    # Show the ERP state in words ("Received") rather than the stored code ("sent").
    shown = shown.assign(erp_status=shown.erp_status.map(lambda s: reg.ERP_LABELS.get(s, s)))
    st.caption(f"Showing {len(shown)} of {len(df)} items. "
               "Edit 'Assigned to' and 'Notes' in the table, then press Save changes.")

    # data_editor is an editable table. Only assigned_to and notes can be edited:
    # the serial and status come from the reader or the review screen, so they
    # stay as they are here.
    edited = st.data_editor(
        shown, hide_index=True, width="stretch",
        disabled=[c for c in reg.ASSET_COLUMNS if c not in ("assigned_to", "notes")],
        column_config={
            "id": st.column_config.NumberColumn("#", width="small"),
            "serial": "Serial / product number",
            "item_type": "Type",
            "status": "Status",
            "erp_status": "ERP",
            "assigned_to": st.column_config.TextColumn("Assigned to"),
            "notes": st.column_config.TextColumn("Notes"),
            "scanned_at": "Scanned",
            "source": "Read from",
        },
        # A new key when the filters change, so the table starts fresh for the new rows.
        key=f"assets_{query}_{statuses}_{types}")

    # Compare the edited table with what was shown: only changed rows are saved.
    before = shown.set_index("id")[["assigned_to", "notes"]]
    after = edited.set_index("id")[["assigned_to", "notes"]].fillna("")
    changed = after[(after != before).any(axis=1)]

    save_col, export_col = st.columns(2)
    if save_col.button(f"Save changes ({len(changed)})", type="primary", disabled=changed.empty):
        reg.update_details([(r.assigned_to.strip(), r.notes.strip(), int(i)) for i, r in changed.iterrows()])
        st.toast(f"Saved {len(changed)} item(s)")
        st.rerun()
    # Exports what's on screen (after filters), e.g. "all laptops assigned to Sam".
    export_col.download_button("Export shown rows (CSV)", shown.to_csv(index=False),
                               file_name="assets.csv", mime="text/csv")


def erp_screen(job_id: str | None, online: bool) -> None:
    st.header("ERP")
    if not online:
        st.error("The ERP isn't reachable. Start it in a second terminal:  `python mock_erp.py serve`")
        return
    if job_id is None:
        st.warning("No job on the ERP yet. Create one:  "
                   "`python mock_erp.py seed --job JOB-0001 --from truth.csv`")
        return

    try:
        lines = reg.erp().get_lines(job_id)
    except ErpError as e:
        st.error(f"Couldn't load {job_id}: {e}")
        return

    # 1. Progress on the job's manifest.
    expected = sum(line["quantity"] for line in lines)
    received = sum(min(line["received"], line["quantity"]) for line in lines)
    c1, c2, c3 = st.columns(3)
    c1.metric("Manifest lines", len(lines))
    c2.metric("Received", f"{received} / {expected}")
    c3.metric("Waiting to send", reg.pending_count(job_id))
    st.progress(received / expected if expected else 0.0)

    with st.expander(f"Manifest of {job_id}", expanded=False):
        st.dataframe(
            [{"Line": line["line_id"], "Identifier": line["identifier"], "Type": line["identifier_type"],
              "Description": line["description"], "Received": f"{line['received']} / {line['quantity']}"}
             for line in lines],
            hide_index=True, width="stretch")

    # 2. Posts waiting in the outbox (the ERP was down or busy when they were made).
    st.subheader("Waiting to send")
    n_pending = reg.pending_count(job_id)
    if n_pending == 0:
        st.caption("Nothing waiting.")
    elif st.button(f"Retry sending ({n_pending})", type="primary"):
        with st.spinner("Sending..."):
            error = reg.retry_pending(job_id)
        if error:
            st.error(f"The ERP still couldn't be reached: {error}")
        else:
            st.toast("All waiting items were answered by the ERP.")
            st.rerun()

    # 3. Accepted / confirmed items this job hasn't seen yet.
    st.subheader("Not sent yet")
    unsent = reg.unsent_items(job_id)
    if not unsent:
        st.caption("Every accepted or confirmed item has been sent to this job.")
    else:
        st.dataframe([{"#": i["id"], "Serial / product number": i["serial"], "Status": i["status"],
                       "ERP": reg.ERP_LABELS.get(i["erp_status"] or "", i["erp_status"])} for i in unsent],
                     hide_index=True, width="stretch")
        if st.button(f"Send these {len(unsent)} to {job_id}"):
            with st.spinner("Sending..."):
                results = [reg.post_item(i["id"], job_id)[0] for i in unsent]
            st.toast(", ".join(f"{results.count(s)} {reg.ERP_LABELS.get(s, s).lower()}"
                               for s in dict.fromkeys(results)))
            st.rerun()

    # 4. What the ERP step couldn't decide on its own.
    st.subheader("Needs a decision")
    attention = reg.attention_items(job_id)
    if not attention:
        st.caption("Nothing.")
    for item in attention:
        with st.container(border=True):
            st.write(f"**Item #{item['id']}** · {item['serial']} · "
                     f"{reg.ERP_LABELS.get(item['erp_status'], item['erp_status'])}")
            st.caption(item["erp_detail"])
            if item["erp_status"] == "near_match":
                # Same choice as on the Review screen: pick the manifest line this really is.
                m = reg.match_preview(item["serial"], job_id)
                for line in m.candidates:
                    if st.button(f"It's line {line['line_id']}: {line['identifier']} "
                                 f"({line['description']}), send it",
                                 key=f"fix_{item['id']}_{line['line_id']}"):
                        reg.correct_serial(item["id"], line["identifier"])
                        status, _, _ = reg.post_item(item["id"], job_id)
                        st.toast(f"Item #{item['id']}: {reg.ERP_LABELS.get(status, status)}")
                        st.rerun()


# ---------- page ----------

st.set_page_config(page_title="Label Reader – Asset Register", layout="wide")
st.sidebar.title("Asset Register")
n_review = len(reg.review_items())
# The options stay fixed and only their displayed text shows the count. If the
# options themselves changed ("Review (3)" -> "Review (2)"), Streamlit would see
# a new widget and jump back to the first screen after every confirm.
page = st.sidebar.radio("Screen", ["Scan", "Review", "Assets", "ERP"], key="page",
                        format_func=lambda p: f"Review ({n_review})" if p == "Review" else p)

# ERP connection and job, shown on every screen.
st.sidebar.divider()
online = reg.erp_online()
job_id = None
if online:
    try:
        jobs = reg.erp().list_jobs()
    except ErpError as e:
        jobs = None
        st.sidebar.error(f"ERP error: {e}")
    if jobs:
        st.sidebar.success("ERP connected")
        job_id = st.sidebar.selectbox(
            "Receiving into job", [j["job_id"] for j in jobs], key="job_id",
            format_func=lambda j: next(f"{j} · {x['received']}/{x['expected']} received"
                                       for x in jobs if x["job_id"] == j))
    elif jobs == []:
        st.sidebar.warning("ERP connected, but it has no jobs yet.")
else:
    st.sidebar.warning("ERP offline. Scans are still saved; send them from the ERP screen later.")

if page == "Scan":
    scan_screen(job_id)
elif page == "Review":
    review_screen(job_id)
elif page == "Assets":
    assets_screen()
else:
    erp_screen(job_id, online)
