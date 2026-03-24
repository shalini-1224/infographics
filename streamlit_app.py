import uuid
from pathlib import Path

import streamlit as st

from main import _setup_logging, run_pipeline


APP_TEMP_ROOT = Path("temp")
UPLOAD_ROOT = APP_TEMP_ROOT / "uploads"
LOG_ROOT = APP_TEMP_ROOT / "logs"
RESULT_ROOT = APP_TEMP_ROOT / "results"


def _save_uploaded_files(uploaded_files):

    batch_id = uuid.uuid4().hex
    batch_dir = UPLOAD_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for uploaded_file in uploaded_files:
        destination = batch_dir / uploaded_file.name
        destination.write_bytes(uploaded_file.getbuffer())
        saved_files.append(destination)

    return batch_id, batch_dir, saved_files


def _render_chart(chart, chart_index):

    st.markdown(f"**Chart {chart_index}**")
    header = chart.get("chart_header", "")
    legends = chart.get("chart_legends", [])
    narrative = chart.get("chart_narrative", "")

    if header:
        st.markdown(f"**Header:** {header}")

    if legends:
        st.markdown("**Legends**")
        for legend in legends:
            st.write(f"- {legend}")

    if chart.get("chart_md"):
        st.markdown("**Chart Table**")
        st.markdown(chart["chart_md"])

    if narrative:
        st.markdown("**Narrative**")
        st.write(narrative)


def _render_pdf_result(result):

    pdf_title = f"{result['file_name']} ({len(result.get('infographic_results', []))} page(s))"

    with st.expander(pdf_title, expanded=False):
        st.caption(result["file_path"])

        infographic_results = sorted(
            result.get("infographic_results", []),
            key=lambda item: item.get("page_number", 0),
        )

        if not infographic_results:
            st.info("No image pages were found in this PDF.")
            return

        for page_result in infographic_results:
            st.markdown(f"### Page {page_result['page_number']}")
            left_col, right_col = st.columns([1, 1.2], gap="large")

            with left_col:
                st.image(
                    page_result["image_path"],
                    caption=f"Page {page_result['page_number']}",
                    use_container_width=True,
                )

            with right_col:
                charts = page_result.get("charts", [])
                if not charts:
                    st.info("No charts extracted for this page.")
                else:
                    for index, chart in enumerate(charts, start=1):
                        _render_chart(chart, index)

            st.divider()


def main():

    st.set_page_config(
        page_title="PDF Infographic Extractor",
        layout="wide",
    )
    st.title("PDF Infographic Extractor")
    st.write("Upload one or more PDF files, run the extraction pipeline, and review the generated JSON, images, markdown tables, and metadata.")

    with st.sidebar:
        st.header("Run Settings")
        max_workers = st.number_input("Max workers", min_value=1, max_value=16, value=4, step=1)
        config_path = st.text_input("Config path", value="config.json")
        run_button = st.button("Run Extraction", type="primary", use_container_width=True)

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if run_button:
        if not uploaded_files:
            st.error("Upload at least one PDF file.")
            return

        batch_id, batch_dir, _ = _save_uploaded_files(uploaded_files)
        log_file = LOG_ROOT / f"{batch_id}.log"
        output_file = RESULT_ROOT / f"{batch_id}.json"

        _setup_logging(str(log_file))

        with st.spinner("Running extraction pipeline..."):
            try:
                final_payload = run_pipeline(
                    input_path=str(batch_dir),
                    temp_root=str(APP_TEMP_ROOT),
                    config_path=config_path,
                    output_path=str(output_file),
                    max_workers=int(max_workers),
                )
            except Exception as error:
                st.error(f"Pipeline failed: {error}")
                if log_file.exists():
                    st.code(log_file.read_text(encoding="utf-8"), language="text")
                return

        st.success("Extraction complete.")

        st.markdown("## Run Log")
        if log_file.exists():
            st.code(log_file.read_text(encoding="utf-8"), language="text")
        else:
            st.info("No log file was generated.")

        st.markdown("## Extracted Pages")
        results = final_payload.get("results", [])
        if not results:
            st.info("No PDF results available.")
            return

        st.caption(f"{len(results)} PDF file(s) processed")
        for result in results:
            _render_pdf_result(result)


if __name__ == "__main__":
    main()
