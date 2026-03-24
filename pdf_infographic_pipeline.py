"""
PDF Infographic Analytics Pipeline
====================================
Steps:
  1. Parse PDF → extract text per page
  2. Detect infographic pages (low text density + visual richness heuristics)
  3. Rasterize all pages to images
  4. Use Claude (Anthropic API) to narrate infographic pages visually
  5. Link infographic narratives to text chunks via LDA topic modelling
  6. Analytics: topic distributions, similarity scores, keyword summaries

Dependencies:
    pip install pypdf pdfplumber pymupdf anthropic gensim nltk scikit-learn
                pillow matplotlib seaborn wordcloud pandas

Usage:
    python pdf_infographic_pipeline.py --pdf your_file.pdf [--output_dir ./output]
"""

import os
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import fitz                         # PyMuPDF
import pdfplumber
import anthropic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from gensim.models import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

for pkg in ("punkt", "stopwords", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

ANTHROPIC_MODEL = "claude-opus-4-5"   # vision-capable model


# ──────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────
@dataclass
class PageData:
    page_num: int                           # 1-based
    text: str = ""
    image_path: Optional[str] = None
    is_infographic: bool = False
    infographic_narrative: str = ""
    text_density: float = 0.0
    image_count: int = 0
    topic_vector: list = field(default_factory=list)
    dominant_topic: int = -1


# ──────────────────────────────────────────────────────────────
# Step 1 – PDF Parsing
# ──────────────────────────────────────────────────────────────
def parse_pdf(pdf_path: str) -> list[PageData]:
    """Extract per-page text and image counts."""
    log.info("Step 1 – Parsing PDF: %s", pdf_path)
    pages: list[PageData] = []

    doc = fitz.open(pdf_path)
    with pdfplumber.open(pdf_path) as plumber_pdf:
        for i, (fitz_page, plumb_page) in enumerate(
                zip(doc.pages(), plumber_pdf.pages), start=1):

            text = plumb_page.extract_text() or ""
            # count embedded raster images
            img_count = len(fitz_page.get_images(full=False))
            # text density = chars / page area  (pt²)
            area = fitz_page.rect.width * fitz_page.rect.height
            density = len(text.strip()) / max(area, 1)

            pages.append(PageData(
                page_num=i,
                text=text,
                image_count=img_count,
                text_density=density,
            ))

    doc.close()
    log.info("  Parsed %d pages", len(pages))
    return pages


# ──────────────────────────────────────────────────────────────
# Step 2 – Infographic Detection
# ──────────────────────────────────────────────────────────────
def detect_infographics(pages: list[PageData],
                        density_threshold: float = 0.002,
                        min_images: int = 1) -> list[PageData]:
    """
    Flag a page as infographic when:
      - text density is below threshold  (visually rich, text-sparse), OR
      - it contains embedded raster images
    Tune thresholds per your document type.
    """
    log.info("Step 2 – Detecting infographic pages …")
    densities = [p.text_density for p in pages]
    # Adaptive threshold: 25th percentile of all densities if no fixed one works
    adaptive = float(np.percentile(densities, 25)) if densities else density_threshold
    thr = min(density_threshold, adaptive)

    count = 0
    for p in pages:
        p.is_infographic = (p.text_density < thr) or (p.image_count >= min_images)
        if p.is_infographic:
            count += 1

    log.info("  Found %d infographic page(s) out of %d", count, len(pages))
    return pages


# ──────────────────────────────────────────────────────────────
# Step 3 – Rasterise pages to images
# ──────────────────────────────────────────────────────────────
def rasterize_pages(pdf_path: str,
                    pages: list[PageData],
                    output_dir: str,
                    dpi: int = 150) -> list[PageData]:
    """Render every page to a JPEG and store the path in PageData."""
    log.info("Step 3 – Rasterising pages to images (dpi=%d) …", dpi)
    img_dir = Path(output_dir) / "page_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    for p in pages:
        fitz_page = doc[p.page_num - 1]
        pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
        img_path = img_dir / f"page_{p.page_num:04d}.jpg"
        pix.save(str(img_path))
        p.image_path = str(img_path)

    doc.close()
    log.info("  Saved images to %s", img_dir)
    return pages


# ──────────────────────────────────────────────────────────────
# Step 4 – LLM narration of infographic pages
# ──────────────────────────────────────────────────────────────
def narrate_infographics(pages: list[PageData],
                         client: anthropic.Anthropic) -> list[PageData]:
    """
    Send each infographic page image to Claude vision and ask it to extract
    the full narrative: data points, labels, trends, and key insights.
    """
    log.info("Step 4 – Narrating infographic pages with Claude …")
    infographic_pages = [p for p in pages if p.is_infographic]
    log.info("  Sending %d page(s) to Claude …", len(infographic_pages))

    for p in infographic_pages:
        if not p.image_path or not Path(p.image_path).exists():
            log.warning("  Page %d: image not found, skipping", p.page_num)
            continue

        with open(p.image_path, "rb") as f:
            img_bytes = f.read()

        import base64
        b64 = base64.standard_b64encode(img_bytes).decode()

        prompt = (
            "You are an expert data analyst. Examine this page from a PDF document.\n\n"
            "Extract a COMPLETE NARRATIVE that includes:\n"
            "1. What type of visual is shown (chart, diagram, infographic, map, etc.)\n"
            "2. All visible text labels, titles, axis labels, legends\n"
            "3. All numerical values and data points\n"
            "4. Key trends, comparisons, or patterns visible\n"
            "5. The main message or insight the visual communicates\n"
            "6. Any contextual information (time periods, categories, units)\n\n"
            "Be specific and comprehensive – this narrative will be used for analytics."
        )

        try:
            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            p.infographic_narrative = resp.content[0].text
            log.info("  Page %d: narration complete (%d chars)",
                     p.page_num, len(p.infographic_narrative))
        except Exception as e:
            log.error("  Page %d: Claude API error – %s", p.page_num, e)
            p.infographic_narrative = f"[Narration failed: {e}]"

    return pages


# ──────────────────────────────────────────────────────────────
# Step 5 – LDA topic modelling + linking
# ──────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> list[str]:
    """Lowercase, tokenise, remove stop-words and short tokens."""
    try:
        stop = set(stopwords.words("english"))
    except LookupError:
        stop = set()

    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and len(t) > 3 and t not in stop]


def build_lda_and_link(pages: list[PageData],
                       num_topics: int = 6,
                       passes: int = 15) -> tuple[models.LdaModel,
                                                   corpora.Dictionary,
                                                   list[PageData]]:
    """
    Build one LDA model over ALL page content (text + infographic narrative).
    Assign topic vectors to every page.
    """
    log.info("Step 5 – Building LDA model (topics=%d, passes=%d) …",
             num_topics, passes)

    docs: list[list[str]] = []
    for p in pages:
        # Combine text + infographic narrative for a richer representation
        combined = p.text + " " + p.infographic_narrative
        docs.append(preprocess_text(combined))

    # Remove empty docs
    docs = [d if d else ["empty"] for d in docs]

    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(d) for d in docs]

    lda = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=42,
        alpha="auto",
        eta="auto",
    )

    # Assign topic vectors to pages
    for i, p in enumerate(pages):
        vec = lda.get_document_topics(corpus[i], minimum_probability=0.0)
        p.topic_vector = [float(prob) for _, prob in
                          sorted(vec, key=lambda x: x[0])]
        p.dominant_topic = int(np.argmax(p.topic_vector)) if p.topic_vector else -1

    # Compute cosine similarity between infographic narratives and text chunks
    infographic_pages = [p for p in pages if p.is_infographic]
    text_pages = [p for p in pages if not p.is_infographic and p.text.strip()]

    if infographic_pages and text_pages:
        inf_vecs = np.array([p.topic_vector for p in infographic_pages])
        txt_vecs = np.array([p.topic_vector for p in text_pages])

        if inf_vecs.shape[1] > 0 and txt_vecs.shape[1] > 0:
            sim_matrix = cosine_similarity(inf_vecs, txt_vecs)
            log.info("  Similarity matrix shape: %s", sim_matrix.shape)

            for i, inf_page in enumerate(infographic_pages):
                top_k = np.argsort(sim_matrix[i])[::-1][:3]
                related = [text_pages[j].page_num for j in top_k]
                log.info("  Infographic p.%d → most related text pages: %s",
                         inf_page.page_num, related)

    log.info("  LDA model built; topics assigned to %d pages", len(pages))
    return lda, dictionary, pages


# ──────────────────────────────────────────────────────────────
# Step 6 – Analytics & Visualisations
# ──────────────────────────────────────────────────────────────
def run_analytics(pages: list[PageData],
                  lda: models.LdaModel,
                  dictionary: corpora.Dictionary,
                  output_dir: str,
                  num_topics: int = 6) -> None:
    """
    Produce:
      A) Per-page topic heatmap
      B) Infographic vs text page comparison
      C) Topic keyword bar charts
      D) Cosine similarity heatmap (infographic ↔ text pages)
      E) JSON summary report
    """
    log.info("Step 6 – Running analytics …")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([{
        "page": p.page_num,
        "is_infographic": p.is_infographic,
        "text_density": p.text_density,
        "image_count": p.image_count,
        "dominant_topic": p.dominant_topic,
        **{f"topic_{t}": (p.topic_vector[t] if t < len(p.topic_vector) else 0.0)
           for t in range(num_topics)},
    } for p in pages])

    topic_cols = [f"topic_{t}" for t in range(num_topics)]

    # ── A) Topic heatmap ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(12, len(pages) // 2), 5))
    heat_data = df[topic_cols].T
    heat_data.columns = [f"p{r}" for r in df["page"]]
    sns.heatmap(heat_data, ax=ax, cmap="YlOrRd", linewidths=0.3,
                cbar_kws={"label": "Topic probability"})
    ax.set_title("Topic Distribution Across Pages", fontsize=14, fontweight="bold")
    ax.set_xlabel("Page")
    ax.set_ylabel("Topic")
    # mark infographic pages
    infographic_cols = [f"p{r}" for r in df.loc[df["is_infographic"], "page"]]
    for col in infographic_cols:
        if col in heat_data.columns:
            idx = list(heat_data.columns).index(col)
            ax.axvline(x=idx, color="blue", linewidth=2, alpha=0.5)
    ax.annotate("│ = infographic page", xy=(0.01, -0.12),
                xycoords="axes fraction", color="blue", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "topic_heatmap.png", dpi=150)
    plt.close(fig)
    log.info("  Saved topic_heatmap.png")

    # ── B) Text density distribution ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in [
        (axes[0], "text_density", "Text Density per Page"),
        (axes[1], "image_count", "Embedded Image Count per Page"),
    ]:
        colors = ["#e74c3c" if v else "#3498db"
                  for v in df["is_infographic"]]
        ax.bar(df["page"], df[col], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Page")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#e74c3c", label="Infographic"),
                            Patch(color="#3498db", label="Text page")])
    fig.tight_layout()
    fig.savefig(out / "page_characteristics.png", dpi=150)
    plt.close(fig)
    log.info("  Saved page_characteristics.png")

    # ── C) Topic keyword charts ───────────────────────────────
    top_n = 8
    topics_data = lda.show_topics(num_topics=num_topics,
                                  num_words=top_n, formatted=False)
    cols_per_row = 3
    rows = (num_topics + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row,
                             figsize=(cols_per_row * 5, rows * 3.5))
    axes = axes.flatten()

    for tid, topic_terms in topics_data:
        words = [w for w, _ in topic_terms]
        scores = [s for _, s in topic_terms]
        axes[tid].barh(words[::-1], scores[::-1], color=plt.cm.tab10(tid / num_topics))
        axes[tid].set_title(f"Topic {tid}", fontweight="bold")
        axes[tid].set_xlabel("Weight")

    for ax in axes[num_topics:]:
        ax.set_visible(False)

    fig.suptitle("LDA Topic Keywords", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "topic_keywords.png", dpi=150)
    plt.close(fig)
    log.info("  Saved topic_keywords.png")

    # ── D) Cosine similarity heatmap ─────────────────────────
    inf_df = df[df["is_infographic"]]
    txt_df = df[~df["is_infographic"] & (df["text_density"] > 0)]

    if not inf_df.empty and not txt_df.empty:
        inf_vecs = inf_df[topic_cols].values
        txt_vecs = txt_df[topic_cols].values
        sim = cosine_similarity(inf_vecs, txt_vecs)

        fig, ax = plt.subplots(figsize=(max(8, len(txt_df) // 2),
                                         max(4, len(inf_df))))
        sns.heatmap(sim, ax=ax,
                    xticklabels=[f"p{r}" for r in txt_df["page"]],
                    yticklabels=[f"infographic p{r}" for r in inf_df["page"]],
                    cmap="Blues", annot=(sim.size <= 100),
                    fmt=".2f", linewidths=0.5,
                    cbar_kws={"label": "Cosine similarity"})
        ax.set_title("Infographic ↔ Text Page Similarity (via LDA)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Text pages")
        ax.set_ylabel("Infographic pages")
        fig.tight_layout()
        fig.savefig(out / "similarity_heatmap.png", dpi=150)
        plt.close(fig)
        log.info("  Saved similarity_heatmap.png")

        # For each infographic, record top-3 related text pages
        linkage = {}
        for i, row in inf_df.iterrows():
            scores = sim[list(inf_df.index).index(i)]
            top3_idx = np.argsort(scores)[::-1][:3]
            linkage[int(row["page"])] = [
                {"text_page": int(txt_df.iloc[j]["page"]),
                 "similarity": float(scores[j])}
                for j in top3_idx
            ]
    else:
        linkage = {}
        log.warning("  Not enough infographic/text pages for similarity heatmap.")

    # ── E) JSON report ────────────────────────────────────────
    topic_summaries = {}
    for tid, terms in topics_data:
        topic_summaries[f"topic_{tid}"] = [w for w, _ in terms]

    report = {
        "total_pages": len(pages),
        "infographic_pages": [p.page_num for p in pages if p.is_infographic],
        "text_pages": [p.page_num for p in pages if not p.is_infographic],
        "lda_topics": topic_summaries,
        "infographic_linkage": linkage,
        "page_details": [
            {
                "page": p.page_num,
                "is_infographic": p.is_infographic,
                "dominant_topic": p.dominant_topic,
                "text_snippet": p.text[:300].strip(),
                "infographic_narrative": p.infographic_narrative[:500]
                    if p.infographic_narrative else "",
            }
            for p in pages
        ],
    }

    report_path = out / "analytics_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("  Saved analytics_report.json")

    # ── Console summary ───────────────────────────────────────
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE – SUMMARY")
    print("═" * 60)
    print(f"  Total pages         : {len(pages)}")
    print(f"  Infographic pages   : {sum(p.is_infographic for p in pages)}")
    print(f"  LDA topics          : {num_topics}")
    print(f"\n  Output files in     : {out.resolve()}")
    for f_name in ["topic_heatmap.png", "page_characteristics.png",
                   "topic_keywords.png", "similarity_heatmap.png",
                   "analytics_report.json"]:
        path = out / f_name
        if path.exists():
            print(f"    ✓ {f_name}")
    print("═" * 60 + "\n")


# ──────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────
def run_pipeline(pdf_path: str,
                 output_dir: str = "./output",
                 num_topics: int = 6,
                 dpi: int = 150,
                 density_threshold: float = 0.002) -> None:

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. "
            "Export it before running:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic()

    # Step 1 – Parse
    pages = parse_pdf(pdf_path)

    # Step 2 – Detect infographics
    pages = detect_infographics(pages, density_threshold=density_threshold)

    # Step 3 – Rasterise
    pages = rasterize_pages(pdf_path, pages, output_dir, dpi=dpi)

    # Step 4 – Narrate infographics
    pages = narrate_infographics(pages, client)

    # Step 5 – LDA + linking
    lda, dictionary, pages = build_lda_and_link(pages, num_topics=num_topics)

    # Step 6 – Analytics
    run_analytics(pages, lda, dictionary, output_dir, num_topics=num_topics)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF Infographic Analytics Pipeline"
    )
    parser.add_argument("--pdf", required=True,
                        help="Path to the input PDF file")
    parser.add_argument("--output_dir", default="./output",
                        help="Directory for all output files (default: ./output)")
    parser.add_argument("--num_topics", type=int, default=6,
                        help="Number of LDA topics (default: 6)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for page rasterisation (default: 150)")
    parser.add_argument("--density_threshold", type=float, default=0.002,
                        help="Text-density threshold for infographic detection "
                             "(default: 0.002; lower = more pages flagged)")

    args = parser.parse_args()
    run_pipeline(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        num_topics=args.num_topics,
        dpi=args.dpi,
        density_threshold=args.density_threshold,
    )
