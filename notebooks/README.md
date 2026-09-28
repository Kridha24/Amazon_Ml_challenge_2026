# Notebooks — Amazon ML Challenge 2026

This directory contains standalone, end-to-end Jupyter and Google Colab notebooks for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

---

## Notebook Overview

| Notebook | Focus & Architecture | Key Innovations |
| :--- | :--- | :--- |
| **`amazon_v6_high_recall.ipynb`** | Candidate Recall Optimization | Multi-pass inverted index blocking, 128D FAISS embeddings, 4-fold OOF cross-validation, and cross-encoder re-ranking. |
| **`amazon_v7_stable_high_recall.ipynb`** | GPU Acceleration & Memory Stability | PyTorch exact kNN blocking on GPU, locked high-recall blocker configuration (`K_NAME=20`, `K_FULL=30`), CPU-hist XGBoost/LightGBM. |
| **`amazon_v8_fast_hard_negative.ipynb`** | Fast Hard-Negative Mining | Two-pass hard-negative mining: retains all true positives, selects hard negatives per entity, trains in 1/3 the time while preserving full honest validation. |
| **`amazon_ml_2026_entity_resolution_v7_colab.ipynb`** | Colab-Tuned Pipeline | Specialized runtime setup for Google Colab environments. |

---

## How to Run

### Option 1: Running in Google Colab (Recommended for GPU)
1. Upload the notebook of your choice (e.g., `amazon_v8_fast_hard_negative.ipynb`) to [Google Colab](https://colab.research.google.com/).
2. Select a GPU runtime: **Runtime > Change runtime type > T4, L4, or A100 GPU**.
3. **Data Setup**:
   - In Section 2 of the notebook, configure `DRIVE_LINK` with your `student_resource.zip` shareable link or file ID, **OR**
   - Upload `student_resource.zip` directly to the `/content/` directory.
4. Run all cells from top to bottom.

### Option 2: Running Locally
1. Ensure dependencies are installed in your virtual environment:
   ```bash
   pip install -r ../requirements.txt
   pip install faiss-cpu  # or faiss-gpu if on CUDA Linux
   ```
2. In Section 1 of the notebook, set `DATA_PATH` to the path of your extracted dataset or `student_resource.zip`:
   ```python
   DATA_PATH = "../dataset"
   ```
3. Start Jupyter:
   ```bash
   jupyter lab
   ```

---

## Core Components in the Pipeline
1. **Multilingual & Indic Address Normalization**: Transliterates Indian script addresses (Tamil, Hindi, Telugu, etc.), expands standard abbreviations, cleans honorific prefixes (`M/s`, `Shri`), and corrects OCR digit substitutions.
2. **Multi-Index Blocking**: Uses token prefixes, sorted-token keys, rare business tokens, and dense embeddings to achieve >96% candidate recall before classification.
3. **Pairwise Feature Engineering**: Computes edit distances, Jaccard token overlap, prefix matches, and numerical/pin-code alignment without relying on source entity IDs.
4. **Classification & Entity-Level Macro F0.5 Tuning**: Tunes decision thresholds specifically to maximize Macro $F_{0.5}$, penalizing false positive entity assignments.
