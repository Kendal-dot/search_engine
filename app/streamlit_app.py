# app/streamlit_app.py
# Streamlit UI for multimodal search (CLIP): Text->Image and Image->Text
# ---------------------------------------------------------------
# Requirements: torch, torchvision, transformers, pillow, pandas, numpy, streamlit
# Paths assume running from project root: `streamlit run app/streamlit_app.py`

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor
import streamlit as st


# --------------------
# Config & Paths
# --------------------
st.set_page_config(page_title="Multimodal Search (CLIP)", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = PROJECT_ROOT / "embeddings"
DATA_DIR = PROJECT_ROOT / "data" / "flickr8k"
IMAGES_DIR = DATA_DIR / "Images"

IMAGE_EMB_FILE = EMB_DIR / "image_embeddings.npy"
META_FILE = EMB_DIR / "metadata.csv"
TEXT_EMB_FILE = EMB_DIR / "text_embeddings.npy"

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------
# Cached loaders
# --------------------
@st.cache_resource(show_spinner=True)
def load_clip(model_name: str = MODEL_NAME):
    """Load CLIP model + processor on chosen device; cached across reruns."""
    model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


@st.cache_data(show_spinner=True)
def load_image_embeddings() -> Tuple[np.ndarray, pd.DataFrame]:
    """Load image embeddings and metadata from Part 1."""
    if not IMAGE_EMB_FILE.exists():
        raise FileNotFoundError(f"Missing {IMAGE_EMB_FILE}. Run Part 1 to generate it.")
    if not META_FILE.exists():
        raise FileNotFoundError(f"Missing {META_FILE}. Run Part 1 to generate it.")
    embs = np.load(IMAGE_EMB_FILE)  # [N, D] L2-normalized
    meta = pd.read_csv(META_FILE)   # columns: image, caption
    if len(meta) != embs.shape[0]:
        raise ValueError("Mismatch: rows in metadata vs embeddings. Re-run Part 1.")
    return embs, meta


@st.cache_data(show_spinner=True)
def build_or_load_text_embeddings(meta: pd.DataFrame, batch_size: int = 128) -> np.ndarray:
    """
    Return L2-normalized text embeddings for meta['caption'].
    If a cached .npy exists, load it; otherwise compute and save.
    """
    # Try load cached file
    if TEXT_EMB_FILE.exists():
        tex = np.load(TEXT_EMB_FILE)
        if tex.shape[0] == len(meta):
            return tex
        # fallback to recompute if mismatch
    # Compute fresh
    model, processor = load_clip()
    captions = meta["caption"].astype(str).tolist()
    all_vecs = []
    with torch.no_grad():
        for start in range(0, len(captions), batch_size):
            chunk = captions[start:start + batch_size]
            inputs = processor(text=chunk, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            feats = model.get_text_features(**inputs)  # [B, D]
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            all_vecs.append(feats.detach().cpu().numpy())
    text_embs = np.concatenate(all_vecs, axis=0)
    np.save(TEXT_EMB_FILE, text_embs)
    return text_embs


# --------------------
# Embedding helpers
# --------------------
@torch.no_grad()
def embed_text(query: str, model: CLIPModel, processor: CLIPProcessor) -> np.ndarray:
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    feats = model.get_text_features(**inputs)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.detach().cpu().numpy()  # [1, D]


@torch.no_grad()
def embed_image(pil_img: Image.Image, model: CLIPModel, processor: CLIPProcessor) -> np.ndarray:
    inputs = processor(images=[pil_img.convert("RGB")], return_tensors="pt").to(DEVICE)
    feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.detach().cpu().numpy()  # [1, D]


def topk_from_scores(scores: np.ndarray, k: int = 5) -> np.ndarray:
    """Return indices of top-k scores descending."""
    k = min(k, scores.shape[0])
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


# --------------------
# UI Components
# --------------------
def sidebar_about():
    st.sidebar.markdown(
        """
        ## About this app
        **Multimodal Search (CLIP)**  
        - **Text → Image**: encode a text query and retrieve the most similar images.  
        - **Image → Text**: upload an image and retrieve the most similar text descriptions.  

        **Tech stack:**  
        - [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) via 🤗 Transformers  
        - Embeddings: cosine similarity on L2-normalized vectors  
        - Data: Flickr8k (images + captions)  
        - UI: Streamlit

        **Files used:**  
        - `embeddings/image_embeddings.npy`  
        - `embeddings/metadata.csv`  
        - `embeddings/text_embeddings.npy` (auto-generated on first run)
        """
    )
    st.sidebar.markdown(
        f"**Device:** `{DEVICE}`  \n"
        f"**Model:** `{MODEL_NAME}`"
    )


def show_image_results(image_names: List[str], meta: pd.DataFrame, scores: List[float]):
    cols = st.columns(len(image_names))
    for col, img_name, score in zip(cols, image_names, scores):
        path = IMAGES_DIR / img_name
        cap = meta.loc[meta["image"] == img_name, "caption"].values[0]
        with col:
            st.image(str(path), caption=f"{img_name}\ncos={score:.3f}", use_column_width=True)
            st.markdown(f"**Caption:** {cap}")


def show_text_results(rows: pd.DataFrame):
    for i, row in rows.iterrows():
        st.markdown(f"**{i+1}. Cosine:** {row['cosine']:.3f}")
        st.write(row["caption"])
        # Also show the associated image for context (optional but nice)
        img_path = IMAGES_DIR / row["image"]
        st.image(str(img_path), caption=row["image"], use_column_width=True)
        st.divider()


# --------------------
# Main App
# --------------------
def main():
    st.title("Multimodal Search Engine — CLIP")
    st.caption("Search images from text, and captions from an uploaded image.")

    sidebar_about()

    # Load data / model
    image_embs, meta = load_image_embeddings()          # [N, D], DataFrame
    model, processor = load_clip()

    tab_text2img, tab_img2text = st.tabs(["🔎 Text → Image", "🖼️ Image → Text"])

    # ------------- Text -> Image -------------
    with tab_text2img:
        st.subheader("Text to Image Search")
        query = st.text_input(
            "Enter a natural language query:",
            value="a brown dog running on the beach"
        )
        topk = st.slider("Top-K results", min_value=1, max_value=10, value=5, step=1)

        demo = st.toggle("Use some demo queries", value=False)
        if demo:
            st.write("Try one of these:")
            demo_cols = st.columns(3)
            demos = [
                "two people riding bicycles in a city",
                "a small child playing with a red ball",
                "a white snow-covered mountain under blue sky",
            ]
            for c, q in zip(demo_cols, demos):
                if c.button(q):
                    query = q

        if st.button("Search", type="primary"):
            if not query.strip():
                st.warning("Please enter a non-empty query.")
            else:
                with st.spinner("Embedding query and retrieving results..."):
                    q_emb = embed_text(query, model, processor)       # [1, D]
                    scores = image_embs @ q_emb[0]                     # [N]
                    idx = topk_from_scores(scores, k=topk)
                    images = meta.loc[idx, "image"].tolist()
                    result_scores = scores[idx]
                st.success("Done!")
                show_image_results(images, meta, result_scores)
                # Simple table of results
                df_out = pd.DataFrame({"image": images, "cosine": result_scores})
                st.dataframe(df_out.reset_index(drop=True), use_container_width=True)

    # ------------- Image -> Text -------------
    with tab_img2text:
        st.subheader("Image to Text Search")
        uploaded = st.file_uploader(
            "Upload an image (jpg, jpeg, png):",
            type=["jpg", "jpeg", "png"]
        )
        topk2 = st.slider("Top-K captions", min_value=1, max_value=10, value=5, step=1, key="topk2")

        # Optionally precompute text embeddings (first run may compute automatically)
        if st.button("Precompute text embeddings (optional)"):
            with st.spinner("Computing text embeddings for all captions..."):
                tex_embs = build_or_load_text_embeddings(meta)
            st.success(f"Text embeddings ready: shape {tex_embs.shape}")

        if uploaded is not None and st.button("Find captions", type="primary"):
            # Ensure we have text embeddings
            with st.spinner("Loading/Computing text embeddings..."):
                text_embs = build_or_load_text_embeddings(meta)  # [N, D]

            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Uploaded image", use_column_width=True)

            with st.spinner("Embedding image and retrieving captions..."):
                img_emb = embed_image(pil_img, model, processor)   # [1, D]
                scores = text_embs @ img_emb[0]                    # [N]
                idx = topk_from_scores(scores, k=topk2)

                rows = meta.loc[idx, ["image", "caption"]].copy()
                rows["cosine"] = scores[idx]
                rows = rows.sort_values("cosine", ascending=False).reset_index(drop=True)

            st.success("Done!")
            show_text_results(rows)

    # Footer
    st.caption(
        "Built with 🤗 Transformers & CLIP. Cosine similarity over L2-normalized embeddings."
    )


if __name__ == "__main__":
    main()
