# FairML: Classification and Fairness Analysis on the COMPAS Dataset

> **Can a recidivism prediction model be both accurate *and* fair?**  
> This project trains five ML classifiers on the COMPAS dataset and uses AIF360 Reweighing to reduce racial bias — without sacrificing predictive performance.

---

##  Project Overview

The COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) tool was used in the US criminal justice system to estimate recidivism risk. ProPublica's 2016 investigation revealed significant racial disparities in its predictions — particularly higher false positive rates for African-American defendants.

This project:
- Trains 5 classification models to predict recidivism (`is_recid`)
- Evaluates them on **both performance and fairness** metrics
- Applies **AIF360 Reweighing** for bias mitigation
- Selects a final model balancing accuracy and fairness

**Authors:** Raveena Kumari & Sneha Pillai — TH Köln, Communication Systems and Networks

---

## Repository Structure

```
# Repository Structure

```text
ML_3/
├── .gitignore                      # Excludes checkpoints and large raw data files
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── FairML.ipynb                    # Main Jupyter notebook (full ML & fairness pipeline)
├── data/
│   ├── compas_clean.csv            # Cleaned & preprocessed COMPAS dataset
│   └── compas_model_encoded.csv    # Encoded dataset ready for modeling
└── results/                        # Generated evaluation metrics (recreated by the notebook)
    ├── accuracy_fairness_tradeoff_analysis.csv
    ├── final_fair_model_summary.csv
    ├── model_results_all_5_models.csv
    ├── model_selection_ranking.csv
    ├── reweighed_model_results.csv
    └── tuned_model_results.csv

│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes checkpoints, large raw files
└── README.md
```

---

## Pipeline

```
Raw COMPAS Data
      │
      ▼
 Preprocessing
 (ordinal encoding, one-hot, race binary, 80/20 stratified split)
      │
      ▼
 COMPAS Baseline Evaluation
 (score_text → binary prediction, fairness error rates by group)
      │
      ▼
 Train 5 Models
 (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM)
      │
      ▼
 Hyperparameter Tuning (GridSearchCV, F1-score metric)
      │
      ▼
 Fairness Evaluation (AIF360 + manual group calculations)
      │
      ▼
 Bias Mitigation — AIF360 Reweighing
      │
      ▼
 Final Model Selection (accuracy + F1 + ROC-AUC + Fairness Gap Score)
```

---

##  Results

### Model Performance (Tuned)

| Model                    | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|--------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression      | 0.6554   | 0.6107    | 0.3037 | 0.4057   | 0.6802  |
| Decision Tree            | 0.6238   | 0.5244    | 0.3055 | 0.3861   | 0.6476  |
| Random Forest            | 0.6403   | 0.5549    | 0.3588 | 0.4358   | 0.6430  |
| Gradient Boosting        | 0.6513   | 0.5773    | 0.3712 | 0.4519   | 0.6768  |

### Fairness Metrics — Original Models

| Model               | FPR Diff. | FNR Diff. | Sel. Diff. | Disparate Impact | Avg. Odds |
|---------------------|-----------|-----------|-----------|-----------------|-----------|
| Logistic Regression | 0.1112    | -0.2486   | 0.1731    | 0.8104          | -0.1799   |
| Decision Tree       | 0.0813    | -0.1947   | 0.1343    | 0.8470          | -0.1380   |
| Random Forest       | 0.1457    | -0.2794   | 0.2056    | 0.7709          | -0.2126   |
| Gradient Boosting   | 0.1487    | -0.2786   | 0.2084    | 0.7649          | -0.2137   |
| SVM                 | 0.1279    | -0.2310   | 0.1742    | 0.8092          | -0.1795   |

### Final Model: Reweighed Gradient Boosting

| Metric              | COMPAS Baseline | Final Model |
|---------------------|-----------------|-------------|
| Accuracy            | 0.5921          | **0.6534**  |
| ROC-AUC             | 0.6034          | **0.6769**  |
| FPR Difference      | 0.2199          | **0.0241**  |
| Selection Rate Diff.| 0.2268          | **0.0530**  |
| Fairness Gap Score  | —               | **0.0492**  |

> The reweighed Gradient Boosting model outperforms the COMPAS baseline on both accuracy **and** fairness.

---

##  Key Concepts

| Term | Definition |
|------|-----------|
| **Protected attribute** | `race_binary` — 0 = African-American (unprivileged), 1 = Caucasian (privileged) |
| **Favorable outcome** | No recidivism (`is_recid = 0`) |
| **FPR Difference** | Gap in false positive rates between racial groups (lower = fairer) |
| **Fairness Gap Score** | Project-defined score = avg of \|FPR Diff\|, \|FNR Diff\|, \|Sel. Diff\| |
| **Reweighing** | AIF360 pre-processing: upweights underrepresented group-label combos |

---

##  Getting Started

## Setup Instructions

Follow these steps to set up the project environment and run the fairness analysis pipeline.

### 1. Clone the Repository

```bash
git clone https://github.com/snehaeengineer-cmyk/fair-ml-compas.git
cd fair-ml-compas
```

### 2. Set Up the Data

Because the original raw dataset (`cox-violent-parsed.csv`) is too large to track in version control, you will need to download it manually.

#### Steps

1. Download the raw dataset from the official source.
2. Create a local folder or keep it in your workspace directory as required by your environment paths.
3. Place the downloaded file in the appropriate data directory used by the notebooks/scripts.

> **Note:** The preprocessed datasets (`compas_clean.csv` and `compas_model_encoded.csv`) are already included in the `data/` directory and can be used directly for model training and evaluation.

### 3. Install Dependencies

It is recommended to use a virtual environment before installing the required packages.

#### Create and Activate a Virtual Environment

**macOS/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

#### Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Run the Analysis

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open and execute the notebooks in the recommended order:

1. Data Cleaning and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Model Training and Evaluation
4. Fairness Assessment and Bias Analysis

```

> Run all cells in order. Charts and result CSVs are generated automatically.

---

## References

1. Angwin et al., "Machine Bias," ProPublica, 2016. https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing  
2. Bellamy et al., "AI Fairness 360," arXiv:1810.01943, 2018.  
3. Kamiran & Calders, "Data preprocessing techniques for classification without discrimination," KAIS, 2012.  
4. Pedregosa et al., "Scikit-learn: Machine learning in Python," JMLR, 2011.

---

## Disclaimer

This is an academic analysis only. Real criminal justice applications require deeper legal, ethical, and domain expert review.

# fair-ml-compas
# fair-ml-compas
