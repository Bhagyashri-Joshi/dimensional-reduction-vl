import streamlit as st

st.title("Theory")
st.caption("Read each tab before running the simulation. Equations are rendered with LaTeX.")

tabs = st.tabs(
    ["High dimensions", "PCA", "SVD", "t-SNE", "UMAP", "Trustworthiness", "Classification & leakage", "Limitations"]
)

with tabs[0]:
    st.markdown(
        """
### High-dimensional data and the curse of dimensionality
The HAR dataset describes every 2.56-second window of smartphone motion with **561 numbers** (means, standard deviations,
energies, correlations, frequency-domain features ...). A sample is therefore a point in $\\mathbb{R}^{561}$.

As the number of dimensions $d$ grows:
- data become **sparse** - the volume of the space grows much faster than the number of samples;
- distances between points **concentrate**, so "near" and "far" become less informative;
- models need more data to avoid **overfitting**, and training and storage cost rise;
- you cannot **look** at the data - humans can plot only 2 or 3 dimensions.

### Feature reduction vs. feature selection
| | Feature **selection** | Feature **extraction / reduction** |
|---|---|---|
| Output | A subset of the *original* columns | *New* features built from all originals |
| Examples | Variance threshold, mutual information | PCA, SVD, t-SNE, UMAP |
| Interpretability | Columns keep their meaning | New axes are mixtures |

This lab studies **feature extraction (dimensionality reduction)**.

### Linear vs nonlinear
- **Linear** (PCA, SVD): the new coordinates are linear combinations $z = W^\\top x$.
- **Nonlinear / manifold learning** (t-SNE, UMAP): assume the data lie near a curved low-dimensional surface and preserve neighbourhood relationships.
"""
    )

with tabs[1]:
    st.markdown(
        """
### Principal Component Analysis (PCA)
1. **Centre** (and usually **standardise**) the data: $\\tilde{x}_{ij} = (x_{ij}-\\mu_j)/\\sigma_j$.
2. Compute the covariance matrix $C = \\tfrac{1}{n-1}\\tilde{X}^\\top\\tilde{X}$.
3. Its eigenvectors $v_1, v_2, \\dots$ (the **principal components**) are orthogonal directions ordered by the variance they capture; eigenvalues $\\lambda_k$ give that variance.
4. Project: $Z = \\tilde{X} V_k$ keeps the first $k$ components.

**Explained variance ratio** of component $k$ and the **cumulative** value:
$$\\text{EVR}_k = \\frac{\\lambda_k}{\\sum_j \\lambda_j},\\qquad \\text{Cum}_k = \\sum_{i\\le k}\\text{EVR}_i$$

Choose $k$ as the smallest value with $\\text{Cum}_k \\ge 0.90$ or $0.95$ (a rule of thumb, not a law).
PCA is **unsupervised** - it never sees activity labels - and can transform new samples with the fitted $V_k$.
"""
    )

with tabs[2]:
    st.markdown(
        """
### Singular Value Decomposition (SVD) and Truncated SVD
Any matrix factorises as
$$X = U\\,\\Sigma\\,V^\\top$$
with orthonormal $U, V$ and non-negative **singular values** $\\sigma_1\\ge\\sigma_2\\ge\\dots$ on the diagonal of $\\Sigma$.
**Truncated SVD** keeps only the first $k$: $X \\approx U_k\\Sigma_k V_k^\\top$, and the reduced data are $Z = X V_k = U_k\\Sigma_k$.

**PCA vs Truncated SVD**
- PCA is the SVD of the **centred** (and scaled) data. Eigenvalues relate to singular values by $\\lambda_k = \\sigma_k^2/(n-1)$.
- scikit-learn's `TruncatedSVD` does **not centre** the data. On uncentred data the first component often just points toward the mean, so results differ from PCA. They are *related*, not identical.
- In this lab PCA receives **standardised** features while Truncated SVD receives the **raw** features so that the difference is visible.
- Large singular values = directions that carry much of the matrix's "energy".
"""
    )

with tabs[3]:
    st.markdown(
        """
### t-SNE (t-distributed Stochastic Neighbor Embedding)
t-SNE turns distances into **neighbour probabilities** and finds a 2D layout whose probabilities match.
In the original space the similarity of $j$ to $i$ is Gaussian, $p_{j|i}\\propto \\exp(-\\lVert x_i-x_j\\rVert^2/2\\sigma_i^2)$;
in the map a heavy-tailed Student-t distribution is used, $q_{ij}\\propto (1+\\lVert y_i-y_j\\rVert^2)^{-1}$.
The map is optimised by gradient descent to minimise the KL divergence
$$\\mathrm{KL}(P\\,\\|\\,Q)=\\sum_{i\\neq j}p_{ij}\\log\\frac{p_{ij}}{q_{ij}}.$$

**Perplexity** sets each $\\sigma_i$ so that the point has roughly that many *effective neighbours* (typical 5-50, always smaller than the number of samples).
Other settings: **learning rate**, **iterations**, **random seed** (different seeds give different layouts).

**Reading a t-SNE plot:** local neighbourhoods are meaningful; **cluster sizes, gaps between clusters and axes are not**.
Standard scikit-learn t-SNE has **no `transform()` for new data**.
"""
    )

with tabs[4]:
    st.markdown(
        """
### UMAP (Uniform Manifold Approximation and Projection)
UMAP builds a weighted **nearest-neighbour graph** of the data (a fuzzy topological representation of the manifold) and then
optimises a low-dimensional layout whose graph is as similar as possible.

- **`n_neighbors`** - size of the local neighbourhood. Small values emphasise fine local structure; large values capture more global structure.
- **`min_dist`** - how tightly points may be packed in the embedding (0 = tight clumps, larger = more spread out).
- **`n_components`** - output dimensionality (2 for plots; 5-50 for downstream models).
- **Seed** - fixes the random initialisation (this also makes UMAP single-threaded).

UMAP is usually faster than t-SNE, often keeps more global layout, and **can `transform()` new samples** after fitting - so it can be used in a train/test classification workflow.
"""
    )

with tabs[5]:
    st.markdown(
        """
### Trustworthiness
Let $U_k(i)$ be the points that are among the $k$ nearest neighbours of $i$ in the **embedding** but not in the **original space**
(false neighbours), and $r(i,j)$ the rank of $j$ among $i$'s original-space neighbours. Then
$$T(k)=1-\\frac{2}{nk(2n-3k-1)}\\sum_{i=1}^{n}\\sum_{j\\in U_k(i)}\\bigl(r(i,j)-k\\bigr).$$

- Range **0 to 1**; **1** means every embedded neighbourhood is a true neighbourhood.
- It measures **local neighbourhood preservation** - *not* class separation, *not* classification accuracy and *not* explained variance.
- It depends on $k$ (must satisfy $k < n/2$). Compare embeddings only with the same $k$ and the same samples.
- Here it is computed against the matrix the algorithm received (standardised features for PCA, t-SNE and UMAP; raw features for Truncated SVD).
"""
    )

with tabs[6]:
    st.markdown(
        """
### Classification performance and data leakage
We compare a **Logistic Regression** model trained on (a) the original 561 features and (b) PCA, Truncated SVD or UMAP representations.

**Metrics** (test set, macro-averaged over the 6 activities so each class counts equally):
$$\\text{Precision}=\\tfrac{TP}{TP+FP},\\quad \\text{Recall}=\\tfrac{TP}{TP+FN},\\quad F_1=\\tfrac{2PR}{P+R}$$
plus **accuracy** and the **confusion matrix** (rows = true activity, columns = predicted).

**Leakage-free protocol used in this lab**
1. Use the dataset's predefined train/test split.
2. Fit the scaler **on training data only**; apply it to test data.
3. Fit PCA / SVD / UMAP **on training data only**; *transform* the test data.
4. Fit Logistic Regression on training representations; evaluate once on the test representation.
5. Never choose parameters using test labels.

Fitting any step on train+test lets test information leak into the model and inflates the score.

**Why t-SNE is separate:** it has no fitted `transform()`, so embedding train and test with it would either leak test data or use a different, non-comparable workflow. It is evaluated by visualisation, trustworthiness and runtime.
"""
    )

with tabs[7]:
    st.markdown(
        """
### Limitations of 2D embeddings
- Squeezing 561 dimensions into 2 **must** lose information; some neighbourhoods will be distorted.
- t-SNE and UMAP axes have **no physical meaning**; cluster sizes and between-cluster distances can be misleading.
- Results depend on **parameters and seeds** - always try several settings.
- Well-separated colours **do not prove** that a classifier will succeed; overlapping colours do not prove it will fail.
- Visualisations in this lab use a **reproducible, class-balanced subset** of the training data for speed; classification uses the full predefined split.
- Scores come from **one run per setting**; small differences may be noise.
"""
    )
