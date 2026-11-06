# Product Requirements Document (PRD)
## Dimensionality Reduction Explorer — Interactive Machine Learning Virtual Laboratory

**Document version:** 1.0  
**Status:** Approved for implementation  
**Product type:** Educational virtual laboratory / Streamlit web application  
**Primary audience:** Undergraduate Computer Engineering students  
**Assessment context:** Individual deployment-based academic assessment (10 marks)  
**Primary experiment:** Dimensionality reduction on the Human Activity Recognition (HAR) dataset  
**Algorithms:** PCA, Truncated SVD, t-SNE, UMAP

---

## 1. Product Overview

Dimensionality Reduction Explorer is an interactive virtual laboratory that enables undergraduate students to study, implement, visualize, and compare dimensionality reduction techniques on a high-dimensional Human Activity Recognition (HAR) dataset.

The lab is designed to emulate the educational flow of established virtual laboratory experiences. It is not intended to be a general-purpose ML platform. Its primary purpose is to provide a structured experiment with learning material, pre-test, guided simulation, procedure, exercises, post-test, references, contributor details, and feedback.

Students will be able to explore how dimensionality reduction transforms data with 561 input features into lower-dimensional representations, examine the visual structure of activity classes, and investigate the effect of reduced feature sets on a classification model.

### 1.1 Product statement

Provide an accessible, browser-based laboratory in which students can conduct a reproducible dimensionality reduction experiment and interpret the trade-offs between information retention, visualization, computational cost, and downstream classification performance.

### 1.2 Problem statement

High-dimensional datasets are difficult to visualize and can increase computational and modeling complexity. Students often learn dimensionality reduction through mathematical descriptions or static diagrams, making it difficult to observe the effect of algorithm choice and parameter changes. A guided, interactive virtual lab can connect theoretical concepts with practical outcomes.

### 1.3 Product goals

1. Teach the principles and practical behavior of PCA, Truncated SVD, t-SNE, and UMAP.
2. Allow students to run these techniques on a common, high-dimensional HAR dataset.
3. Provide meaningful visualizations and quantitative evaluation.
4. Compare baseline classification performance with reduced-feature performance for PCA, SVD, and UMAP using a common methodology.
5. Evaluate t-SNE separately for visualization and neighborhood preservation.
6. Provide an IIT Kharagpur Virtual Labs-inspired educational structure.
7. Deploy a working application accessible through a public Streamlit Community Cloud URL.

### 1.4 Non-goals

The initial release will not:
- Support user-uploaded datasets.
- Provide user accounts, authentication, or persistent student progress.
- Offer a general-purpose notebook environment.
- Provide real-time sensor streaming or live activity recognition.
- Train deep neural networks or learned feature embeddings.
- Claim that a visualization proves class separability or causality.
- Compare t-SNE classification accuracy as if it had the same out-of-sample transform workflow as PCA, SVD, and UMAP.
- Include an administrative content management system.

---

## 2. Target Users and Personas

### 2.1 Primary persona — Undergraduate student

**Needs**
- Understand the experiment objective and theory.
- Follow a defined procedure.
- Operate simulation controls without coding.
- Observe plots and metrics.
- Complete pre-test, exercises, and post-test.
- Interpret and report experimental results.

**Success outcome:** The student can explain why dimensionality reduction is used, distinguish the four algorithms, and interpret the resulting metrics.

### 2.2 Evaluator — Faculty member

**Needs**
- Verify that the deployed application implements the stated experiment.
- Inspect educational sections and simulation behavior.
- Confirm that results and methodology are technically credible.
- Review the deployment and demonstrate the experiment.

**Success outcome:** The deployed application presents a complete, coherent, and demonstrable virtual lab suitable for academic evaluation.

---

## 3. Learning Outcomes

After completing the virtual lab, a student should be able to:

1. Explain the purpose and limitations of dimensionality reduction.
2. Distinguish linear techniques (PCA and SVD) from nonlinear manifold-learning techniques (t-SNE and UMAP).
3. Apply PCA and Truncated SVD to a high-dimensional dataset.
4. Configure and interpret t-SNE and UMAP embeddings.
5. Interpret PCA explained variance and SVD singular values.
6. Evaluate embedding neighborhood preservation using trustworthiness.
7. Compare classification performance using original features and reduced representations for PCA, SVD, and UMAP.
8. Identify the effect of dimensionality, parameters, and runtime on results.
9. Draw evidence-based conclusions from plots and metrics.

---

## 4. Dataset Requirements

### 4.1 Primary dataset

**Human Activity Recognition Using Smartphones (UCI HAR Dataset)**

- Input features: 561 engineered features.
- Classes: 6 activities.
- Activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, and LAYING.
- Dataset is based on smartphone accelerometer and gyroscope measurements collected from participants.
- The dataset's predefined train/test partitions should be used when available to preserve the benchmark split.

### 4.2 Dataset use

The dataset is used for:
- Feature matrix input to dimensionality reduction.
- Activity labels for color-coded visualization.
- Baseline classification.
- Reduced-feature classification.
- Trustworthiness analysis.

### 4.3 Data handling requirements

- Provide a deterministic dataset loading path.
- Clearly identify the source and license/citation in the References section.
- Check that data and labels are aligned.
- Handle missing or unavailable dataset files with a visible actionable message.
- Avoid silently changing the dataset or its class labels.
- Display dataset dimensions and class distribution.

---

## 5. Product Scope and Information Architecture

The app shall use a visible navigation sidebar or navigation menu. The following pages/sections are in scope:

1. **Home / Overview**
2. **Aim**
3. **Theory**
4. **Pre-test**
5. **Simulation**
6. **Procedure**
7. **Observations and Results**
8. **Exercises**
9. **Post-test**
10. **References**
11. **Contributors**
12. **Feedback**

### 5.1 Home / Overview
- Show experiment title, brief introduction, course outcome mapping (CO1, CO3), and learning objectives.
- Provide a clear route to begin the experiment.
- Display the experiment workflow.

### 5.2 Aim
Present the defined aim:

> To apply PCA, SVD, t-SNE, and UMAP to the high-dimensional Human Activity Recognition dataset, visualize reduced feature representations, analyze information and neighborhood preservation, and evaluate the effect of reduced feature sets on classification performance.

### 5.3 Theory
Include student-appropriate explanations of:
- High-dimensional data and the curse of dimensionality.
- Feature reduction versus feature selection.
- PCA: centering, principal components, explained variance.
- SVD: matrix factorization and singular values.
- t-SNE: local neighborhood visualization and perplexity.
- UMAP: manifold learning, neighborhood size, and minimum distance.
- Trustworthiness and its interpretation.
- Classification performance and data leakage.
- Limitations of 2D embeddings.

Mathematical expressions may be included in rendered Markdown/LaTeX.

### 5.4 Pre-test
- Provide approximately 10 MCQs.
- Cover fundamental concepts and algorithm distinctions.
- Allow the student to submit answers and receive a score.
- Display correct answers and brief explanations after submission.
- Do not require account creation.

### 5.5 Simulation
The Simulation page is the core product feature.

**Student inputs**
- Dataset: HAR (fixed in MVP; optional supported dataset choices only if implementation time permits).
- Algorithm: PCA, Truncated SVD, t-SNE, UMAP.
- Reduced dimensions: 2, 5, 10, 20, or 50 where valid.
- Algorithm-specific parameters:
  - PCA: component count; optional whitening only if implemented and explained.
  - Truncated SVD: component count.
  - t-SNE: perplexity, learning rate, iteration limit, random seed.
  - UMAP: `n_neighbors`, `min_dist`, component count, random seed.
- Run button.
- Reset/default controls.

**Simulation output**
- Original feature dimension and reduced dimension.
- Selected algorithm and parameters.
- 2D scatter visualization when output dimension is 2.
- 3D visualization only if included and tested; not required for MVP.
- Runtime.
- Appropriate algorithm-specific metrics.
- Brief interpretation notes.
- Clear errors for invalid settings or unsupported combinations.

**Algorithm-specific output**
- PCA: explained variance ratio per component and cumulative explained variance.
- Truncated SVD: singular values; variance-related values only where correctly defined and computed.
- t-SNE: embedding scatter plot and trustworthiness; no explained variance claim.
- UMAP: embedding scatter plot and trustworthiness; no explained variance claim.

### 5.6 Classification comparison

The app shall provide a comparison experiment using Logistic Regression.

**Primary comparison**
- Baseline: Logistic Regression on the original 561-feature HAR input.
- PCA + Logistic Regression.
- Truncated SVD + Logistic Regression.
- UMAP + Logistic Regression.

**Metrics**
- Accuracy.
- Precision (macro).
- Recall (macro).
- F1-score (macro).
- Confusion matrix, at least for the selected model or comparison runs.
- Runtime for reduction and classification, clearly distinguished if both are measured.

**Methodology**
- Use the predefined HAR train/test split when available.
- Fit any preprocessing and dimensionality reduction on training data only.
- Transform test data using the fitted reducer.
- Fit the classifier on reduced training data and evaluate on reduced test data.
- Use the same test samples and common Logistic Regression settings for the baseline and PCA/SVD/UMAP comparisons.
- Use a fixed random seed for reproducibility.
- Avoid fitting transformations using test labels or test data.

**t-SNE treatment**
- t-SNE is evaluated separately as a visualization and neighborhood-preservation method.
- Report embedding visualization, trustworthiness, and execution time.
- Do not include standard t-SNE in the main train/test classification accuracy table because standard scikit-learn t-SNE does not provide the same out-of-sample `transform()` workflow as PCA and UMAP.
- If a t-SNE classification extension is later implemented, it must be labeled as a distinct methodology and documented clearly.

### 5.7 Procedure
Provide numbered instructions:
1. Open the experiment and read the aim and theory.
2. Complete the pre-test.
3. Inspect the HAR dataset details and class distribution.
4. Select an algorithm and valid parameters.
5. Run the simulation.
6. Inspect the reduced embedding and metrics.
7. Run the classification comparison.
8. Compare results and record observations.
9. Complete the exercises and post-test.
10. Review references and submit feedback.

### 5.8 Observations and Results
- Display the selected algorithm and parameter values.
- Display computed results and relevant plots.
- Provide a results table for comparison runs.
- Provide concise interpretation guidance.
- Allow CSV export of result tables if included in MVP.
- Never hard-code fabricated experimental scores.

### 5.9 Exercises
Include guided tasks such as:
- Compare PCA with 2 and 20 components.
- Investigate the cumulative explained variance of PCA.
- Change the number of SVD components.
- Observe t-SNE output under different perplexity values.
- Change UMAP `n_neighbors` and `min_dist`.
- Compare classification metrics before and after PCA/SVD/UMAP.
- Explain why t-SNE is evaluated differently.

### 5.10 Post-test
- Provide approximately 10 MCQs.
- Include applied questions based on simulation observations.
- Show score and explanations after submission.

### 5.11 References
- Cite the UCI HAR dataset.
- Cite original or foundational papers for PCA, SVD, t-SNE, and UMAP as appropriate.
- Cite official scikit-learn and UMAP documentation.
- Use consistent bibliographic formatting.

### 5.12 Contributors
Show contributor name(s), role(s), and faculty guide if provided. Do not invent names, designations, or institutional affiliations.

### 5.13 Feedback
- Provide a short optional feedback form.
- Collect rating and text feedback.
- For MVP, feedback may be displayed as a form without persistent storage if no backend is configured.
- Clearly state if feedback is not saved.

---

## 6. Functional Requirements

| ID | Requirement | Priority | Acceptance criteria |
|---|---|---|---|
| FR-01 | Educational navigation | Must | All in-scope sections are accessible from the navigation UI. |
| FR-02 | Display aim and learning outcomes | Must | Aim and outcomes are visible and specific to the experiment. |
| FR-03 | Display theory | Must | Theory covers all four algorithms and key evaluation concepts. |
| FR-04 | Load HAR dataset | Must | Dataset is loaded and its dimensions and class distribution are displayed. |
| FR-05 | Run PCA | Must | PCA produces a reduced representation and explained variance outputs. |
| FR-06 | Run Truncated SVD | Must | SVD produces a reduced representation and singular values. |
| FR-07 | Run t-SNE | Must | t-SNE produces a reproducible embedding under valid parameters. |
| FR-08 | Run UMAP | Must | UMAP produces a reproducible embedding under valid parameters. |
| FR-09 | Visualize embeddings | Must | Plot shows embedding coordinates and HAR activity labels. |
| FR-10 | Calculate trustworthiness | Must | Trustworthiness is calculated and labeled for supported embeddings. |
| FR-11 | Baseline classification | Must | Logistic Regression baseline is evaluated on the predefined test set. |
| FR-12 | Reduced-feature classification | Must | PCA, SVD, and UMAP comparisons use train-only fitting and the same test split. |
| FR-13 | Show comparison metrics | Must | Accuracy, macro precision, macro recall, and macro F1 are displayed. |
| FR-14 | Pre-test and post-test | Must | Students can answer, submit, and view scores and explanations. |
| FR-15 | Procedure and exercises | Must | Procedure is actionable and exercises include parameter-based tasks. |
| FR-16 | References and contributors | Must | References are credible; contributor details are editable source content. |
| FR-17 | Runtime and status feedback | Must | User sees progress/status and runtime for computational operations. |
| FR-18 | Handle invalid parameters | Must | Invalid settings are prevented or produce helpful error messages. |
| FR-19 | Reproducibility | Must | Random seeds are available and used where supported. |
| FR-20 | Feedback section | Should | Feedback UI is present and communicates whether submissions are stored. |
| FR-21 | Download results | Should | Result table can be downloaded as CSV. |
| FR-22 | Report generation | Could | Generate a concise experiment report from actual session results. |

---

## 7. User Stories

1. As a student, I want to read the aim and theory so that I understand the experiment before running it.
2. As a student, I want to complete a pre-test so that I can assess my prior understanding.
3. As a student, I want to select a dimensionality reduction algorithm and parameters so that I can investigate its behavior.
4. As a student, I want to see a color-coded embedding so that I can visually inspect activity-class structure.
5. As a student, I want to view PCA explained variance and SVD singular values so that I can interpret information represented by components.
6. As a student, I want to compare baseline and reduced-feature classification so that I can understand the effect of dimensionality reduction on predictive performance.
7. As a student, I want to change t-SNE and UMAP parameters so that I can study how embeddings respond.
8. As a student, I want to complete exercises and a post-test so that I can reinforce learning.
9. As an evaluator, I want to navigate the lab and run an experiment so that I can verify the deployed implementation.
10. As an evaluator, I want to inspect the methodology and references so that I can assess the scientific reliability of the experiment.

---

## 8. Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-01 | Usability | Clear navigation and student-friendly language. |
| NFR-02 | Responsiveness | Works on current desktop browsers; layout remains usable on tablet/mobile. |
| NFR-03 | Performance | Common interaction should complete in a reasonable time on Streamlit Cloud; provide status/progress for expensive runs. |
| NFR-04 | Reliability | Invalid settings and missing data are handled without an unhandled app crash. |
| NFR-05 | Reproducibility | Fixed seeds are applied wherever the algorithm supports them. |
| NFR-06 | Scientific correctness | Metrics and claims are methodologically accurate; no fabricated values. |
| NFR-07 | Accessibility | Controls have labels; color-coded classes are also explained with a legend. |
| NFR-08 | Maintainability | Educational content, algorithm logic, and UI are separated into modules. |
| NFR-09 | Deployment | Application is deployable from a GitHub repository to Streamlit Community Cloud. |
| NFR-10 | Cost | MVP uses free/open-source packages and free-tier deployment. |

---

## 9. MVP and Prioritization

### Must-have (MVP)
- Streamlit app with all core educational sections.
- HAR data loader and dataset information.
- PCA, Truncated SVD, t-SNE, UMAP implementations.
- 2D embedding visualization.
- PCA explained variance and SVD singular-value display.
- Trustworthiness metric.
- Baseline versus PCA/SVD/UMAP Logistic Regression comparison.
- Pre-test, procedure, exercises, post-test, references, contributors.
- Error handling and reproducible parameters.
- Public deployment URL.

### Should-have
- CSV download.
- Model/experiment result comparison table.
- Feedback UI.
- Session state for preserving experiment settings and results.

### Could-have
- PDF experiment report.
- 3D embedding visualization.
- Additional datasets.
- Student progress persistence.
- Advanced visual explanations.

### Won't-have in the initial release
- Accounts or authentication.
- Database-backed user records.
- User dataset upload.
- Live sensor input.
- Real-time model serving.

---

## 10. Acceptance Criteria for Assessment

The product is ready for demonstration when:

1. The public deployed URL opens successfully.
2. The app contains the specified educational sections.
3. The HAR dataset loads and reports its correct feature and class information.
4. PCA, Truncated SVD, t-SNE, and UMAP each execute for valid parameters.
5. Each algorithm produces a meaningful visualization.
6. PCA explained variance and SVD singular values are shown correctly.
7. Trustworthiness is calculated for supported embeddings and explained.
8. The baseline and PCA/SVD/UMAP classification comparison is executable without data leakage.
9. Results use actual computed values and are reproducible where supported.
10. Pre-test and post-test scoring work.
11. The app handles invalid parameter combinations with helpful messages.
12. References and contributor information are present.
13. The simulation can be demonstrated within a reasonable time.

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| UMAP package/deployment dependency issues | Pin compatible versions in requirements; test deployment before assessment. |
| t-SNE runtime | Use a reasonable sample count, parameter bounds, and caching where appropriate. |
| Streamlit Cloud memory or CPU limitations | Avoid repeatedly recomputing the full pipeline; cache dataset loading; constrain user controls. |
| Confusion about t-SNE classification | Clearly separate its visualization evaluation from train/test classification comparison. |
| Data leakage in classification | Fit scaling and dimensionality reduction only on training data. |
| Visualization appears clustered but metrics disagree | Explain that 2D visualization is a projection and does not guarantee classifier performance. |
| Dataset download failure | Bundle a legally permitted dataset copy or provide robust loading/error guidance; verify repository size and licensing. |
| Unverified contributor or reference details | Use user-provided names and checked bibliographic sources only. |

---

## 12. Success Measures

The initial release is successful if:
- It is publicly deployed and usable for the assessment.
- Students can complete the experiment flow without writing code.
- The four algorithms are functional and their outputs are understandable.
- The comparison methodology is scientifically defensible.
- The evaluator can verify implementation and educational completeness during a live demonstration.

---

## 13. Approval

This PRD defines the approved product scope for the initial implementation. Any change to the dataset, algorithm evaluation methodology, or major feature scope should be documented before implementation.
