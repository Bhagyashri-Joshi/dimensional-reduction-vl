# Dimensionality Reduction Explorer
An interactive machine-learning virtual laboratory (Streamlit) in which students apply **PCA, Truncated SVD, t-SNE and UMAP**
to the **UCI Human Activity Recognition (HAR)** dataset (561 features, 6 activities), visualise the embeddings, measure
neighbourhood preservation (trustworthiness) and compare Logistic Regression on original vs reduced features.
Structure follows an IIT Kharagpur Virtual Labs-style flow: Overview, Aim, Theory, Pre-test, Simulation, Procedure,
Observations & Results, Exercises, Post-test, References, Contributors, Feedback.

**Deployment URL:** _add your Streamlit Community Cloud URL here after deploying_

## Learning objectives
Explain purpose/limits of dimensionality reduction - distinguish linear vs nonlinear methods - apply PCA/SVD - configure and interpret t-SNE/UMAP -
read explained variance and singular values - use trustworthiness - compare classifiers on reduced features without data leakage.
(Course outcomes CO1, CO3 - edit wording in `src/content.py` if needed.)

## Introduction: ML concepts and technical approach
Machine learning (ML) uses data to find patterns and make predictions or descriptions without writing a separate rule for every case. This virtual lab (VL) explores **dimensionality reduction**: converting data with many features (measurements) into a smaller set of features. This can make data easier to visualize and may reduce computation, but it can also discard useful information.

The lab uses the UCI Human Activity Recognition (HAR) dataset. Each example represents a smartphone motion window described by 561 features, and its label identifies one of six activities. The data are split into training and test sets. A scaler and each reduction method are fitted using training data only; the fitted transformations are then applied to test data. This prevents **data leakage**, where information from the test set influences model training or evaluation.

Four reduction methods are explored:
- **PCA (Principal Component Analysis)** is a linear method that creates orthogonal directions capturing as much variance as possible. Its explained-variance ratios help describe how much variance is retained.
- **Truncated SVD** also finds a lower-dimensional linear representation using singular values. Unlike PCA, the implementation here does not centre its input, so it can produce different results.
- **t-SNE** is a nonlinear visualization method designed to preserve local neighborhoods in a low-dimensional map. Its plots can reveal local groupings, but distances between clusters and the axes should not be interpreted as reliable global measurements. The standard t-SNE implementation used here cannot transform new samples, so it is evaluated separately from classification.
- **UMAP** builds a neighborhood graph and optimizes a lower-dimensional representation. Its settings control the neighborhood scale and how tightly points can be packed; it supports transforming new samples after fitting.

The lab uses **trustworthiness** to measure whether nearby points in an embedding were also neighbors in the original feature space. It also compares **Logistic Regression**, a supervised classifier, on original and reduced features. Accuracy and macro-averaged precision, recall and F1 summarize classification performance, while a confusion matrix shows which activities are mistaken for one another. These measures answer different questions: a visually clear plot does not necessarily produce the best classifier, and no single metric captures every desirable property.

## Features
- All 12 educational sections in a sidebar navigation; ~10-question pre-test and post-test with scores and explanations.
- Simulation: algorithm-specific controls, input validation with actionable errors, run button (no recompute on widget change), reset, seeds, runtime, spinner/status.
- Plots: colour- and marker-coded activity scatter, PCA explained variance (+ full cumulative curve), SVD singular values, metric bars, confusion matrices.
- Classification comparison: Baseline vs PCA / Truncated SVD / UMAP + Logistic Regression (accuracy, macro precision/recall/F1, confusion matrix, reduction vs classifier time).
- t-SNE evaluated separately (visualisation, trustworthiness, runtime, parameters).
- Observations page with result tables, CSV download and a Markdown report generated from *actual* session results. No score is hard-coded anywhere.

## Evaluation methodology
1. Predefined HAR train/test split (7,352 / 2,947). 2. `StandardScaler` and every reducer are fitted on **train only**; test data are only transformed.
3. One Logistic Regression configuration (C=1.0, lbfgs, max_iter=1000) for all models; convergence warnings are reported.
4. Embedding visualisations use a reproducible, class-balanced subset of the training split (default 2,000; t-SNE capped at 3,000).
5. PCA/t-SNE/UMAP receive standardised features; **Truncated SVD receives raw (uncentred) features** so it differs from PCA, as in the TRD.
6. Trustworthiness (`sklearn.manifold.trustworthiness`) is computed against the matrix the algorithm received, with a validated `k < n/2`.
7. UMAP in the comparison can be fitted on a stratified subset of the *training* set (default 3,000) for speed; the full training set and test set are then only transformed. Choose "Full training set" for the strict full-data variant.
8. Limitations: single run per setting; 2D plots can mislead; t-SNE has no out-of-sample transform so it is excluded from the classification table.

## Conclusion
Dimensionality reduction offers a practical way to explore and work with high-dimensional data, but each method preserves different structure. PCA and Truncated SVD provide linear representations; t-SNE focuses on local structure for visualization; UMAP represents neighborhoods and can also transform new samples. Their outputs should therefore be interpreted according to each method's purpose and limitations.

There is no universally best reduction method or setting. A sound comparison considers retained variance where applicable, neighborhood trustworthiness, classification metrics, runtime and the intended use of the reduced data. By fitting preprocessing and reducers on training data only and evaluating on held-out test data, this VL demonstrates how to make that comparison while avoiding data leakage.

## Dataset
UCI HAR Dataset - Anguita et al., ESANN 2013 - https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
(check the page for current license/citation terms). The data are **not committed** to this repo. On first launch the app downloads and
caches them in `data/`; alternatives are in `data/README.md`.

## Local setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
pytest                                                   # 56 tests, synthetic data, no network needed
```
First launch needs internet to fetch the dataset (or place it manually as described in `data/README.md`).

## Deploy to Streamlit Community Cloud
1. Push this folder to a GitHub repository (`app.py` at the repo root). 2. https://share.streamlit.io -> *New app* -> select repo, branch, `app.py`.
3. Advanced settings -> choose Python 3.11 or 3.12. 4. Deploy, then open **every page** and run **all four algorithms and the comparison** on the public URL.
5. Paste the URL into this README. If the cloud download of the dataset is blocked, commit a verified copy under `data/UCI HAR Dataset/` (after confirming redistribution terms).
Tip: open the app once and run the comparison before the demo - results are cached afterwards. The first UMAP run is slow (numba compilation).

## Project structure
```
app.py                  entry point: page config, navigation, session state
pages/                  one file per lab section (UI only)
src/data_loader.py      download, validate and load HAR (predefined split)
src/preprocessing.py    scaling, reproducible stratified sampling
src/dimensionality_reduction.py   PCA / SVD / t-SNE / UMAP wrappers, validation, trustworthiness
src/evaluation.py       leakage-free Logistic Regression comparison, metrics
src/visualization.py    Plotly figures        src/interpretation.py  notes generated from real results
src/content.py         theory-adjacent content, quizzes, references, CONTRIBUTORS (edit!)
src/quiz.py, report.py, state.py, ui.py       quiz scoring, tables/CSV/report, session state, cached calls
tests/                  pytest suite (unit + Streamlit AppTest integration; synthetic data is test-only)
docs/                   PRD and TRD
```

## Before submission - checklist
- [ ] Fill `CONTRIBUTORS`, `FACULTY_GUIDE` (and optionally `COURSE_OUTCOMES` text) in `src/content.py` - the app never invents names.
- [ ] Verify references/license wording against the originals.
- [ ] Deploy, run every algorithm on the public URL, add the URL above.
- [ ] Rehearse the demo path: Overview -> Pre-test -> Simulation (PCA, t-SNE, UMAP) -> Comparison -> Observations -> Post-test.

## Contributors
See the in-app Contributors page (`src/content.py`).
