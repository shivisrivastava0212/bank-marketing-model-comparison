# 🏦 Bank Marketing Strategy: Classification Battle (LR vs KNN vs SVM)

### 📋 Project Overview
This repository implements an industry-standard classification pipeline to predict bank term deposit subscriptions. By comparing **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Support Vector Machines (SVM)**, this project demonstrates how to navigate high-dimensional banking data and address class imbalance to drive marketing efficiency.

The project follows a **Golden Training Pipeline**, ensuring that the models are evaluated on realistic performance metrics rather than just simple accuracy.

---

### 🏗️ Systematic Workflow

#### 1. Data Processing & Realistic Cleaning
* **Data Leakage Prevention:** Following UCI guidelines, I explicitly dropped the `duration` column. Since call duration is unknown *before* a call is made, removing it ensures the model is truly predictive and "deployment-ready."
* **Feature Engineering:** Automated One-Hot Encoding was used to transform categorical demographics (job, education, marital status) into numeric formats.
* **Feature Scaling:** Implemented `StandardScaler` to ensure all features operate on the same scale—a **mechanical necessity** for distance-sensitive models like KNN and SVM.
* **Stratified Split:** Used an 80-20 stratified split to maintain the representative proportion of subscribers vs. non-subscribers in both training and testing sets.

#### 2. The Model Battle
To identify the most robust predictor, three distinct mathematical architectures were evaluated:
* **Baseline (Logistic Regression):** A probabilistic approach to establish a linear performance benchmark.
* **Instance-Based (KNN):** Leveraged local neighborhood patterns ($k=5$) for classification.
* **Margin-Based (SVM):** Optimized with an **RBF Kernel** to detect complex, non-linear boundaries in the feature space.

---

### 📊 Performance Metrics (Final Results)

The models were evaluated using a comprehensive suite of metrics. For this imbalanced dataset, the **F1-Score** was prioritized as the primary KPI:

| Metric | Logistic Regression (Winner) | SVM (RBF) | KNN (k=5) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | ~89.2% | ~88.9% | ~87.5% |
| **Precision (Yes)** | **0.65** | 0.61 | 0.52 |
| **Recall (Yes)** | **0.25** | 0.21 | 0.22 |
| **F1-Score (Yes)** | **0.3617** | **0.3206** | **0.3110** |

#### **Visual Insights**

* **Confusion Matrix:** High True Negatives (TN) indicate the models are excellent at identifying non-subscribers. The "Winner" (Logistic Regression) shows the best balance in identifying "Yes" cases without excessive false alarms.
  <img width="1274" height="343" alt="Screenshot 2026-05-10 at 12 00 02 PM" src="https://github.com/user-attachments/assets/917da7a9-19ae-44f1-aacd-e1018d6c18be" />

* **Scaling Logic:** Internal testing showed that without `StandardScaler`, KNN and SVM accuracy dropped significantly, confirming the importance of normalization.

---

### ⚙️ Configuration & Hyperparameters
Following the **Industry Checklist**, the following parameters provided the most stable results:
* **Algorithm:** Logistic Regression (Optimized with LBFGS solver).
* **SVM Kernel:** RBF (Radial Basis Function) with `C=1.0`.
* **KNN Neighbors:** 5 (Selected to balance bias and variance).
* **Early Stopping Logic:** While these specific models don't use epochs like Neural Networks, the SVM was trained on a specific sample size to prevent computational "overfitting" and ensure stable weights.

---

### 🚀 Deployment & Running Instructions

#### **Option 1: Google Colab (Recommended)**
1. Open the `SVM_+_KNN_Comparison.ipynb` in Colab.
2. Run the first cell; it will prompt you to upload the dataset.
3. Upload `bank-full.csv`.
4. The pipeline will automatically handle cleaning, scaling, and visualization.

#### **Option 2: Local Environment**
1. Clone the repo: `git clone https://github.com/shivisrivastava0212/bank-marketing-classification-battle.git`
2. Install dependencies: `pip install pandas scikit-learn seaborn matplotlib`
3. Ensure the `.csv` file is in the same directory as the script.
4. Run: `python bank_marketing_analysis.py`

---

### 🧠 Lessons Learned
* **Metric Selection:** I learned that in banking, **F1-Score** is far more valuable than Accuracy. A model that predicts "No" for everyone is 88% accurate but 0% useful.
* **Distance Sensitivity:** Observed first-hand how algorithms like KNN and SVM fail without proper standardization, unlike tree-based models.
* **Realistic Constraints:** Removing the `duration` feature taught me to look for "leaky" variables that wouldn't be available in a real-time inference scenario.

---

### 👨‍💻 Author
**Shivi Srivastava** Aspiring AI Engineer | Amity University Uttar Pradesh  
[LinkedIn Profile](https://www.linkedin.com/in/shivi-srivastava-8a5086310/) | [GitHub Portfolio](https://github.com/shivisrivastava0212)
