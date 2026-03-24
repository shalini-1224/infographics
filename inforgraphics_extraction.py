import json
import os
from pathlib import Path
from typing import List

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class ChartData(BaseModel):
    chart_type: str = Field(description="Type of visual such as bar, line, pie, table, map, infographic, or unknown")
    chart_header: str = Field(default="", description="Main visible title or heading")
    chart_legends: List[str] = Field(default_factory=list, description="Legend labels, color labels, or series names")
    chart_md: str = Field(default="", description="Markdown table representation of the chart converted into a structured table")
    chart_narrative: str = Field(default="", description="Detailed explanation of the chart and the visible data")


class InfographicExtractionResult(BaseModel):
    page_number: int
    image_path: str
    charts: List[ChartData] = Field(default_factory=list)


def _load_gemini_api_key(config_path="config.json"):

    if os.getenv("GEMINI_API_KEY"):
        return os.getenv("GEMINI_API_KEY")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Gemini config file not found: {config_path}")

    with config_file.open("r", encoding="utf-8") as file:
        config = json.load(file)

    api_key = config.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing in config.json")

    os.environ["GEMINI_API_KEY"] = api_key
    return api_key


def infographics_extraction(image_path, page_number, config_path="config.json"):

    _load_gemini_api_key(config_path=config_path)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    with open(image_path, "rb") as file:
        image_bytes = file.read()

    prompt = """
    Analyse the chart or infographic in this image.

    Instructions:
    - Identify every chart, table, or infographic panel visible on the page.
    - Read labels, axes, legends, values, units, periods, and category names.
    - Convert the visible visual data into a structured markdown table in `chart_md`.
    - Write a detailed narrative for each chart.
    - If a field is not visible, return an empty value.
    - Return valid JSON only.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png",
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=InfographicExtractionResult,
        ),
    )

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, InfographicExtractionResult):
        result = parsed
    elif isinstance(parsed, dict):
        result = InfographicExtractionResult.model_validate(parsed)
    else:
        result = InfographicExtractionResult.model_validate_json(response.text)

    if result.page_number != page_number:
        result.page_number = page_number
    if result.image_path != image_path:
        result.image_path = image_path

    return result
