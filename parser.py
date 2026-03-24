import os
from pathlib import Path

import fitz  # PyMuPDF


def find_images_in_pdf(pdf_path):

    doc = fitz.open(pdf_path)
    image_pages = {}

    for page_number in range(len(doc)):
        page = doc[page_number]
        image_list = page.get_images(full=True)

        if image_list:
            image_pages[page_number] = []

        for item in image_list:
            try:
                bbox = page.get_image_bbox(item, transform=False)
                image_pages[page_number].append(
                    {
                        "xref": item[0],
                        "bbox": bbox,
                    }
                )
            except Exception as error:
                print(f"Error in page {page_number + 1}: {error}")

    doc.close()
    return image_pages


def build_image_output_dir(pdf_path, temp_root="temp"):

    pdf_name = Path(pdf_path).stem
    output_dir = Path(temp_root) / pdf_name / "img"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_image_pages_as_images(pdf_path, image_pages, output_folder, zoom=2):

    os.makedirs(output_folder, exist_ok=True)

    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(zoom, zoom)
    saved_images = []

    for page_number in sorted(image_pages):
        page = doc[page_number]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
        pix.save(img_path)
        saved_images.append(
            {
                "page_number": page_number + 1,
                "image_path": img_path,
            }
        )

    doc.close()
    return saved_images


def save_pages_as_images(pdf_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    doc = fitz.open(pdf_path)

    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_path = os.path.join(output_folder, f"page_{i + 1}.png")
        pix.save(img_path)

    doc.close()
