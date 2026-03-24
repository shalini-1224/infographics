import argparse
import concurrent.futures
import json
import logging
import os
import time
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from inforgraphics_extraction import InfographicExtractionResult, infographics_extraction
from parser import build_image_output_dir, find_images_in_pdf, save_image_pages_as_images


LOGGER = logging.getLogger(__name__)
RATE_LIMIT_WAIT_SECONDS = 30


class PdfExtractionOutput(BaseModel):
    file_name: str
    file_path: str
    image_pages: List[int] = Field(default_factory=list)
    extracted_images: List[str] = Field(default_factory=list)
    infographic_results: List[InfographicExtractionResult] = Field(default_factory=list)


def _get_pdf_files(input_path):

    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == ".pdf":
        return [path]

    if path.is_dir():
        return sorted(file for file in path.iterdir() if file.suffix.lower() == ".pdf")

    raise FileNotFoundError(f"No PDF file or folder found at: {input_path}")


def _is_rate_limit_error(error):

    message = str(error).lower()
    return any(
        marker in message
        for marker in ("rate limit", "ratelimit", "429", "resource exhausted", "too many requests")
    )


def _extract_with_retry(image_path, page_number, config_path, retry_wait_seconds=RATE_LIMIT_WAIT_SECONDS):

    while True:
        try:
            return infographics_extraction(
                image_path=image_path,
                page_number=page_number,
                config_path=config_path,
            )
        except Exception as error:
            if not _is_rate_limit_error(error):
                raise

            LOGGER.warning(
                "Rate limit detected for page %s (%s). Waiting %s seconds before retrying.",
                page_number,
                image_path,
                retry_wait_seconds,
            )
            time.sleep(retry_wait_seconds)


def process_pdf(pdf_path, temp_root="temp", config_path="config.json"):

    LOGGER.info("Processing PDF: %s", pdf_path)
    image_pages_map = find_images_in_pdf(str(pdf_path))
    LOGGER.info("Found %s image page(s) in %s", len(image_pages_map), pdf_path.name)

    image_output_dir = build_image_output_dir(str(pdf_path), temp_root=temp_root)
    saved_images = save_image_pages_as_images(
        str(pdf_path),
        image_pages=image_pages_map.keys(),
        output_folder=str(image_output_dir),
    )
    LOGGER.info("Saved %s page image(s) under %s", len(saved_images), image_output_dir)

    infographic_results = []
    for saved_image in saved_images:
        LOGGER.info(
            "Sending page %s image to Gemini: %s",
            saved_image["page_number"],
            saved_image["image_path"],
        )
        infographic_results.append(
            _extract_with_retry(
                image_path=saved_image["image_path"],
                page_number=saved_image["page_number"],
                config_path=config_path,
            )
        )
        LOGGER.info("Completed extraction for page %s", saved_image["page_number"])

    return PdfExtractionOutput(
        file_name=pdf_path.name,
        file_path=str(pdf_path.resolve()),
        image_pages=[page_number + 1 for page_number in sorted(image_pages_map.keys())],
        extracted_images=[saved_image["image_path"] for saved_image in saved_images],
        infographic_results=infographic_results,
    )


def process_pdf_files(pdf_files, temp_root="temp", config_path="config.json", max_workers=None):

    pdf_files = list(pdf_files)
    if not pdf_files:
        return []

    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)

    results = []
    if len(pdf_files) <= 1:
        return [process_pdf(pdf_path, temp_root=temp_root, config_path=config_path) for pdf_path in pdf_files]

    worker_count = max(1, min(max_workers, len(pdf_files)))
    LOGGER.info("Processing PDFs in parallel with %s worker(s)", worker_count)

    future_to_pdf = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        for pdf_path in pdf_files:
            future = executor.submit(
                process_pdf,
                pdf_path,
                temp_root,
                config_path,
            )
            future_to_pdf[future] = pdf_path

        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                result = future.result()
                results.append(result)
                LOGGER.info("Finished PDF: %s", pdf_path.name)
            except Exception:
                LOGGER.exception("Failed processing PDF: %s", pdf_path.name)
                raise

    results.sort(key=lambda item: item.file_name)
    return results


def run_pipeline(input_path, temp_root="temp", config_path="config.json", output_path="temp/final_results.json", max_workers=None):

    pdf_files = _get_pdf_files(input_path)
    LOGGER.info("Discovered %s PDF file(s)", len(pdf_files))
    results = process_pdf_files(
        pdf_files,
        temp_root=temp_root,
        config_path=config_path,
        max_workers=max_workers,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_payload = {"results": [result.model_dump(mode="json") for result in results]}

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(final_payload, file, indent=2, ensure_ascii=False)

    LOGGER.info("Saved final JSON output to %s", output_file)
    return final_payload


def _setup_logging(log_file):

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="fwdltrealtyrequirementdemoscheduledfor20thmarch",
        help="Path to a PDF file or a folder containing PDF files",
    )
    parser.add_argument(
        "--temp-root",
        default="temp",
        help="Root folder used to save image pages",
    )
    parser.add_argument(
        "--output",
        default="temp/final_results.json",
        help="Path for the final JSON output",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json containing GEMINI_API_KEY",
    )
    parser.add_argument(
        "--log-file",
        default="temp/main.log",
        help="Path to the execution log file",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Maximum number of PDFs to process in parallel",
    )
    args = parser.parse_args()
    _setup_logging(args.log_file)

    LOGGER.info("Starting PDF infographic extraction")
    LOGGER.info("Input path: %s", args.input)
    LOGGER.info("Output path: %s", args.output)
    LOGGER.info("Log file: %s", args.log_file)
    LOGGER.info("Max workers: %s", args.max_workers)

    final_payload = run_pipeline(
        input_path=args.input,
        temp_root=args.temp_root,
        config_path=args.config,
        output_path=args.output,
        max_workers=args.max_workers,
    )
    print(json.dumps(final_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
