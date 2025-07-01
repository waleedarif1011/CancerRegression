# Cancer Death Rate Regression Analysis

Welcome! This project demonstrates a complete data science workflow for predicting cancer death rates using regression models in Python. The notebook guides you through data loading, cleaning, exploratory analysis, feature engineering, model training, and evaluation—all with clear, step-by-step explanations.

---

## 📋 Project Overview

- **Goal:** Predict the `target_deathrate` (cancer death rate) for US counties using demographic and health-related features.
- **Approach:** Data is loaded, merged, cleaned, explored, and then used to train and evaluate several regression models.
- **Tech Stack:** Python, pandas, numpy, matplotlib, seaborn, scikit-learn.

---

## 🔍 Analysis & Machine Learning

This project is both **analysis and machine learning (ML) based**:
- **Analysis:** Includes exploratory data analysis (EDA), visualization of missing values, feature distributions, and correlation analysis to understand the data and inform modeling decisions.
- **Machine Learning:** Implements a full ML pipeline—feature engineering, model selection, training, evaluation, and hyperparameter tuning—to predict cancer mortality rates.

---

## 🚀 Workflow Steps

1. **Import Libraries**
   - Essential libraries like `numpy`, `pandas`, `matplotlib`, `seaborn`, and scikit-learn modules are imported.
   - Warnings are suppressed and a consistent plot style is set for clarity.

2. **Load and Merge Datasets**
   - Cancer and household datasets are loaded using `pd.read_csv`.
   - Merged on the `geography` column to enrich the feature set.

3. **Data Overview & Missing Values**
   - Use `df.info()` and `df.describe()` for a quick overview.
   - Missing values are checked and visualized with `sns.heatmap`.

4. **Target Distribution**
   - The distribution of `target_deathrate` is visualized using `sns.histplot` to check for skewness and spread.

5. **Correlation Analysis**
   - Correlations between features and the target are explored using `df.corr()` and visualized with a heatmap.

6. **Top Correlated Features**
   - The most important features correlated with `target_deathrate` are identified and visualized with `sns.scatterplot`.

7. **Handle Missing Data**
   - Columns with more than 50% missing values are dropped using `df.drop(columns=drop_cols)`.

8. **Impute Remaining Missing Values**
   - Remaining missing numeric values are filled with the median using `df.fillna(df.median(numeric_only=True))`.

9. **Drop Unnecessary Features**
   - Non-numeric or irrelevant columns like `binnedinc` and `geography` are removed.

10. **Final Data Check**
    - Data integrity is confirmed with `df.info()` and a final check for missing values.

11. **Prepare Features and Target**
    - Features (`X`) and target (`y`) are separated.
    - Features are scaled using `StandardScaler` for better model performance.

12. **Train-Test Split**
    - Data is split into training and testing sets using `train_test_split`.

13. **Model Training & Evaluation**
    - Multiple regression models (`LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `RandomForestRegressor`, `GradientBoostingRegressor`) are trained and evaluated using `r2_score`, `mean_squared_error`, and `mean_absolute_error`.

14. **Model Tuning**
    - The best model (e.g., `RandomForestRegressor`) is tuned using `GridSearchCV` for hyperparameter optimization.

---

## 📈 Results

- The notebook prints and displays the performance of each model.
- The best model is further optimized and its performance is reported.

---

## 🛠️ How to Use

1. **Clone this repository** and ensure you have the required datasets in the `dataset/` folder:
   - `cancer_reg.csv`
   - `avg-household-size.csv`
2. **Install dependencies** (if not already installed):
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. **Open the notebook** (`cancer-regression.ipynb`) in VS Code or Jupyter Notebook.
4. **Run each cell step by step** to follow the workflow and see the results.

---

## 💡 Notes

- All important code and variables are highlighted in markdown cells (e.g., `df`, `target_deathrate`, `train_test_split`).
- Each step is explained with clear bullet points and helpful comments.
- The workflow is modular and easy to adapt for similar regression tasks.

---

## 📚 References

- [scikit-learn documentation](https://scikit-learn.org/)
- [pandas documentation](https://pandas.pydata.org/)
- [seaborn documentation](https://seaborn.pydata.org/)

---
