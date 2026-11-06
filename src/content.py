"""Educational content, kept separate from UI and algorithm code (NFR-08).

Edit CONTRIBUTORS below before the assessment. Nothing in this file contains experimental results.
"""
from __future__ import annotations

from .quiz import Question

EXPERIMENT_TITLE = "Dimensionality Reduction Explorer"
EXPERIMENT_SUBTITLE = "Interactive virtual laboratory on the Human Activity Recognition (HAR) dataset"

# --------------------------------------------------------------------------- contributors (edit me)
# Fill in real details only. Leave the list empty to show a notice on the Contributors page.
# Example entry: {"name": "Your Name", "role": "Developer", "affiliation": "Your department, college"}
CONTRIBUTORS: list[dict[str, str]] = [{"name": "Bhagyashri Joshi", "role": "Developer", "affiliation": "D12B, CMPN, VESIT"}]
FACULTY_GUIDE: dict[str, str] = {"name": "Prof. Ms. Priyanka Shah", "designation": "Assistant Professor", "department": "Computer Engineering"} # e.g. {"name": "Prof. ...", "designation": "...", "department": "..."}

# Course outcomes are mapped by label only; add the official wording if your course sheet requires it.
COURSE_OUTCOMES: dict[str, str] = {"CO1": "Understand the basics of Machine Learning, its types, and essential concepts", "CO3": "Implement supervised learning models for regression and classification problems."}

LEARNING_OUTCOMES = [
    "Explain the purpose and limitations of dimensionality reduction.",
    "Distinguish linear techniques (PCA, SVD) from nonlinear manifold-learning techniques (t-SNE, UMAP).",
    "Apply PCA and Truncated SVD to a high-dimensional dataset.",
    "Configure and interpret t-SNE and UMAP embeddings.",
    "Interpret PCA explained variance and SVD singular values.",
    "Evaluate neighbourhood preservation using trustworthiness.",
    "Compare classification performance on original features and on PCA / SVD / UMAP representations.",
    "Identify the effect of dimensionality, parameters and runtime on results.",
    "Draw evidence-based conclusions from plots and metrics.",
]

AIM = (
    "To apply PCA, SVD, t-SNE, and UMAP to the high-dimensional Human Activity Recognition dataset, "
    "visualize reduced feature representations, analyze information and neighborhood preservation, "
    "and evaluate the effect of reduced feature sets on classification performance."
)

WORKFLOW = [
    ("Learn", "Read the Aim and Theory"),
    ("Pre-test", "Check prior knowledge"),
    ("Explore", "Inspect the HAR dataset"),
    ("Simulate", "Run PCA, SVD, t-SNE, UMAP"),
    ("Compare", "Baseline vs reduced features"),
    ("Reflect", "Observations, exercises, post-test"),
]

PROCEDURE = [
    ("Open the experiment and read the Aim and Theory.", "Use the sidebar. Note the difference between linear (PCA, SVD) and nonlinear (t-SNE, UMAP) methods."),
    ("Complete the Pre-test.", "Submit your answers, then read the explanations for any you missed."),
    ("Inspect the HAR dataset details and class distribution.", "On the Simulation page open *Dataset information*. Note the 561 features, 6 activities and the predefined train/test split."),
    ("Select an algorithm and valid parameters.", "Use *Embedding & visualization*. Only controls relevant to the chosen algorithm are shown; invalid settings are rejected with a message."),
    ("Run the simulation.", "Press **Run algorithm**. Identical settings are cached, so repeating them is instant."),
    ("Inspect the reduced embedding and metrics.", "Check the scatter plot, the algorithm-specific plot (variance / singular values), trustworthiness and runtime."),
    ("Run the classification comparison.", "Open *Classification comparison*, choose the number of reduced dimensions and UMAP settings, then press **Run comparison**."),
    ("Compare results and record observations.", "Open *Observations & Results*, review tables and plots, download the CSV and write your own notes."),
    ("Complete the Exercises and Post-test.", "Exercises ask you to change parameters and explain what you see; the post-test checks your understanding."),
    ("Review References and submit Feedback.", "Complete the Google Form on the Feedback page."),
]

EXERCISES = [
    {
        "title": "Exercise 1 - PCA with 2 versus 20 components",
        "task": "Run PCA with 2 components, then with 20. Note the cumulative explained variance each time.",
        "steps": ["Simulation -> Embedding & visualization -> PCA -> components = 2 -> Run.", "Repeat with components = 20 (only the first two dimensions are plotted).", "Record the cumulative explained variance and trustworthiness for each run."],
        "questions": ["How much variance do 2 components retain compared with 20?", "Does the 2D scatter plot change when more components are computed? Why or why not?"],
    },
    {
        "title": "Exercise 2 - Cumulative explained variance of PCA",
        "task": "Use the full cumulative-variance plot to find how many components are needed for 90%, 95% and 99% of the variance.",
        "steps": ["Run PCA with any component count.", "Read the 90/95/99% lines on the 'all components' plot and the interpretation notes."],
        "questions": ["How many of the 561 features' worth of information is needed for 95% variance?", "Why can a large reduction in dimension still keep most of the variance?"],
    },
    {
        "title": "Exercise 3 - Number of SVD components",
        "task": "Change the Truncated SVD component count (for example 2, 10, 50) and watch the singular values.",
        "steps": ["Run Truncated SVD with 2 components, then 10, then 50.", "Compare the singular-value bar plots."],
        "questions": ["How does the first singular value compare with the rest? Why, given the data are not centred?", "How do the PCA variance plot and SVD singular-value plot differ in what they tell you?"],
    },
    {
        "title": "Exercise 4 - t-SNE perplexity",
        "task": "Run t-SNE with perplexity 5, 30 and 50 (keep the seed and sample size fixed).",
        "steps": ["Select t-SNE, set perplexity = 5, Run.", "Repeat with 30 and 50.", "Compare the plots, trustworthiness and runtime."],
        "questions": ["How do cluster shapes change as perplexity grows?", "Does higher trustworthiness always mean a 'better-looking' plot?"],
    },
    {
        "title": "Exercise 5 - UMAP n_neighbors and min_dist",
        "task": "Vary n_neighbors (5, 15, 50) and min_dist (0.0, 0.1, 0.5) one at a time.",
        "steps": ["Select UMAP and change only n_neighbors; Run each setting.", "Reset and change only min_dist; Run each setting.", "Record trustworthiness and describe the plots."],
        "questions": ["Which parameter controls how tightly points are packed?", "Which parameter trades local detail against broader structure?"],
    },
    {
        "title": "Exercise 6 - Classification before and after reduction",
        "task": "Run the classification comparison with 2, 10 and 50 dimensions.",
        "steps": ["Classification comparison -> dimensions = 2 -> Run comparison.", "Repeat for 10 and 50.", "Open Observations & Results and compare accuracy, macro F1 and times across runs."],
        "questions": ["How does accuracy change relative to the 561-feature baseline as dimensions shrink?", "Which method gives the best trade-off between accuracy and number of features? Is the difference large enough to matter?"],
    },
    {
        "title": "Exercise 7 - Why is t-SNE evaluated differently?",
        "task": "Explain in your own words why t-SNE is not in the classification table.",
        "steps": ["Reread the 'Limitations' part of Theory.", "Look at the t-SNE visualization analysis table in Observations & Results."],
        "questions": ["Which scikit-learn method do PCA, SVD and UMAP provide that t-SNE does not?", "What could go wrong if you fitted t-SNE on train and test together and then classified?"],
    },
]

# --------------------------------------------------------------------------- quizzes
PRE_TEST: list[Question] = [
    Question(
        "What does the 'curse of dimensionality' refer to?",
        ("Data become dense and easy to separate as features are added", "As dimensions grow, data become sparse and distances less informative, so learning gets harder", "Algorithms always run faster in high dimensions", "Only categorical features cause problems"),
        1,
        "In high dimensions the volume grows so quickly that samples become sparse and distances between points concentrate, which hurts distance-based methods and increases overfitting risk.",
    ),
    Question(
        "How does feature *extraction* (e.g. PCA) differ from feature *selection*?",
        ("Extraction keeps a subset of the original columns; selection builds new combined features", "Extraction builds new features from combinations of originals; selection keeps a subset of original features", "They are the same thing", "Extraction needs labels; selection never does"),
        1,
        "PCA creates new axes (linear combinations of all features). Feature selection keeps some of the original columns unchanged.",
    ),
    Question(
        "PCA chooses its first principal component as the direction that...",
        ("Best separates the class labels", "Maximises the variance of the projected data", "Minimises the number of features", "Is closest to the first original feature"),
        1,
        "PCA is unsupervised: it ignores labels and finds orthogonal directions of maximal variance.",
    ),
    Question(
        "Why are features usually standardised before PCA?",
        ("Otherwise PCA cannot run", "Features with large scales would dominate the variance and the components", "Standardising makes the data nonlinear", "To make labels numeric"),
        1,
        "PCA works on variance; a feature measured in large units would dominate the leading components unless scales are made comparable.",
    ),
    Question(
        "Which statement about Truncated SVD in scikit-learn is correct?",
        ("It always centres the data first, so it equals PCA", "It does not centre the data, so it differs from PCA unless the data are already centred", "It is a nonlinear method", "It cannot be applied to dense matrices"),
        1,
        "TruncatedSVD factorises the matrix as given. PCA is SVD of the *centred* data. Without centring the first component often just tracks the mean.",
    ),
    Question(
        "In t-SNE, what does *perplexity* roughly control?",
        ("The number of output dimensions", "The effective number of neighbours each point considers", "The number of classes", "The random seed"),
        1,
        "Perplexity is a smooth measure of the effective neighbourhood size used when converting distances to probabilities. Typical values are 5-50 and it must be smaller than the number of samples.",
    ),
    Question(
        "Increasing UMAP's `n_neighbors` generally...",
        ("Makes the embedding focus more on very local structure", "Makes the embedding balance in more global structure", "Changes the number of output dimensions", "Has no effect"),
        1,
        "Small n_neighbors emphasises fine local structure; large values let UMAP see broader neighbourhoods.",
    ),
    Question(
        "Which methods provide a fitted `transform()` for new, unseen samples in this lab's workflow?",
        ("PCA, Truncated SVD and UMAP", "Only t-SNE", "None of them", "Only PCA"),
        0,
        "PCA, TruncatedSVD and UMAP can transform new data after fitting. Standard scikit-learn t-SNE only offers fit_transform, so it is evaluated separately.",
    ),
    Question(
        "Trustworthiness of an embedding measures...",
        ("Classification accuracy of the embedding", "How well local neighbourhoods are preserved (1 is best)", "The percentage of variance explained", "The runtime of the algorithm"),
        1,
        "Trustworthiness penalises points that appear as neighbours in the embedding but were not neighbours in the original space. It lies between 0 and 1.",
    ),
    Question(
        "Which is an example of data leakage?",
        ("Fitting the scaler on the training set only", "Fitting PCA on train+test data before evaluating a classifier on the test data", "Using a fixed random seed", "Reporting macro-F1"),
        1,
        "Fitting any transformation using test data lets test information influence the model. Fit on train only, then transform the test set.",
    ),
]

POST_TEST: list[Question] = [
    Question(
        "Cumulative explained variance for a PCA run is [0.40, 0.62, 0.74, 0.81, 0.96]. What is the smallest number of components that keeps at least 95%?",
        ("3", "4", "5", "2"),
        2,
        "Cumulative variance first reaches >= 0.95 at the fifth component (0.96).",
    ),
    Question(
        "In your t-SNE plot the activity clusters look well separated. What can you validly conclude?",
        ("A classifier will reach 100% accuracy", "Local neighbourhoods in the data group by activity; the plot alone does not prove classifier performance or tell you true distances between clusters", "The distance between clusters equals their dissimilarity", "PCA must be worse"),
        1,
        "t-SNE preserves local structure; global distances, cluster sizes and axes are not reliable, and a visual impression is not a performance measure.",
    ),
    Question(
        "Embedding A has trustworthiness 0.98 and embedding B has 0.86 (same k). Which statement is justified?",
        ("A preserves local neighbourhoods better than B", "B always gives better classification", "A has more explained variance", "A used fewer samples"),
        0,
        "Higher trustworthiness means fewer false neighbours were introduced. It says nothing directly about variance or classification accuracy.",
    ),
    Question(
        "You increase t-SNE perplexity from 5 to 50 with everything else fixed. What is the most likely effect?",
        ("Each point considers more neighbours, giving a smoother view of broader structure; very small perplexity tends to fragment clusters", "Runtime becomes zero", "The embedding becomes 3D", "Labels change"),
        0,
        "Larger perplexity widens the effective neighbourhood. Always compare several values rather than trusting one plot.",
    ),
    Question(
        "Lowering UMAP `min_dist` from 0.5 to 0.0 mainly...",
        ("Allows points to be packed more tightly in the embedding", "Removes the need for n_neighbors", "Increases the number of dimensions", "Makes UMAP linear"),
        0,
        "min_dist is the minimum allowed distance between embedded points; smaller values give tighter clumps, larger values spread points out.",
    ),
    Question(
        "Why is t-SNE reported in a separate table from the PCA/SVD/UMAP classification comparison?",
        ("t-SNE is too slow to run", "Standard scikit-learn t-SNE has no fitted transform for unseen test samples, so the same train-only-fit workflow is not possible", "t-SNE cannot be plotted", "t-SNE needs labels"),
        1,
        "Without an out-of-sample transform, embedding train and test together would let test data influence the representation (leakage).",
    ),
    Question(
        "In the comparison, a reduced-feature model scores slightly lower accuracy than the 561-feature baseline but uses far fewer features. How should you interpret this?",
        ("The reduction is invalid", "It is a trade-off: some information was discarded in exchange for a much smaller representation; check whether the gap is meaningful", "Accuracy can never differ", "The test set was used for training"),
        1,
        "Dimensionality reduction trades information for compactness. Judge it against the baseline, the cost saved, and run-to-run variability.",
    ),
    Question(
        "Singular values of a Truncated SVD run decrease quickly: the first is much larger than the rest. This suggests...",
        ("The first direction captures much more of the matrix's energy than later ones", "The SVD failed", "All components are equally important", "The data are standardised"),
        0,
        "Singular values measure the importance of each component; for uncentred data a very large first value often reflects the mean.",
    ),
    Question(
        "A student fits UMAP on all 10,299 samples, then splits into train/test for the classifier. What is wrong?",
        ("Nothing", "The reducer saw test samples during fitting, which is data leakage", "UMAP cannot take 10,299 samples", "The classifier must be linear"),
        1,
        "The reducer must be fitted on the training split only; test samples must only be transformed.",
    ),
    Question(
        "Why report *macro* precision/recall/F1 for the six HAR activities?",
        ("They are faster to compute", "Macro-averaging weights every class equally, so weak performance on smaller or harder classes is not hidden", "Macro is always higher than accuracy", "It removes the need for a test set"),
        1,
        "Macro averages compute the metric per class and average them equally. Accuracy can hide poor performance on individual classes.",
    ),
]

# --------------------------------------------------------------------------- references
REFERENCES = {
    "Dataset": [
        "Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A public domain dataset for human activity recognition using smartphones. *Proceedings of the 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2013)*, 437-442.",
        "UCI Machine Learning Repository. *Human Activity Recognition Using Smartphones*. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones (check the dataset page for current license terms and citation).",
    ],
    "Foundational papers": [
        "Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*, 2(11), 559-572.",
        "Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. *Journal of Educational Psychology*, 24(6), 417-441.",
        "Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202.",
        "Golub, G. H., & Reinsch, C. (1970). Singular value decomposition and least squares solutions. *Numerische Mathematik*, 14(5), 403-420.",
        "Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217-288.",
        "van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605.",
        "McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for dimension reduction. *arXiv:1802.03426*.",
        "Venna, J., & Kaski, S. (2001). Neighborhood preservation in nonlinear projection methods: an experimental study. *Artificial Neural Networks - ICANN 2001*, LNCS 2130, 485-491.",
        "Wattenberg, M., Viegas, F., & Johnson, I. (2016). How to use t-SNE effectively. *Distill*. https://distill.pub/2016/misread-tsne/",
    ],
    "Software documentation": [
        "Pedregosa, F., et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.",
        "scikit-learn documentation: PCA - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html",
        "scikit-learn documentation: TruncatedSVD - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html",
        "scikit-learn documentation: TSNE - https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html",
        "scikit-learn documentation: trustworthiness - https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html",
        "scikit-learn documentation: LogisticRegression - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        "UMAP documentation (umap-learn) - https://umap-learn.readthedocs.io/",
        "Streamlit documentation - https://docs.streamlit.io/",
    ],
}
