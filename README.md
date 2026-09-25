# Employee Retention Analysis

Employee retention analysis: predicting which employees are likely to leave, using the well-known IBM Watson HR attrition dataset. This was a mini project (the repo also includes the project documentation PDF and the final presentation), and the notebook compares several classifiers to find the best predictor of attrition.

## Dataset

`data/WA_Fn-UseC_-HR-Employee-Attrition.csv` — 1,470 employees with 35 columns: demographics (age, gender, marital status, distance from home), job details (department, job role, job level, monthly income, overtime, business travel, years at company, years in current role), satisfaction and work-life balance scores, and the target `Attrition` (Yes/No).

## Approach

The notebook (`NOtebook.ipynb`) works through:

1. **Data exploration.** Load the data, check shapes, dtypes, missing values, and summary statistics.
2. **EDA and visualization.** Countplots and boxplots of attrition against key features (overtime, job satisfaction, monthly income, age, etc.) plus a correlation heatmap to spot the strongest relationships.
3. **Preprocessing.** Label-encode the categorical columns, split into train/test sets, and scale the features with `StandardScaler`.
4. **Modeling.** Train and compare classifiers with scikit-learn:
   - Random Forest
   - Decision Tree
   - K-Nearest Neighbors (with GridSearchCV tuning)
   - MLP Classifier (with GridSearchCV tuning)
   - Voting classifier (ensemble of the above)

## Results

Test-set accuracy from the notebook:

- Random Forest: 82.88%
- K-Nearest Neighbors: 81.25%
- Decision Tree: 76.63%
- MLP Classifier: 57.88%
- Voting Classifier: 57.88%

Random Forest was the clear winner. The notebook also compares ROC scores across the models. (Note: the voting classifier's grid search was interrupted mid-run in the notebook, so its score just mirrors the MLP's.)

## How to run

```bash
git clone https://github.com/suhasaitham22/employee-retention.git
cd employee-retention
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook NOtebook.ipynb
```

The notebook expects the CSV under `data/`. There is also a pre-rendered `NOtebook.html` if you just want to read through the analysis.

## Tech stack

Python, pandas, matplotlib, seaborn, scikit-learn, Jupyter Notebook.
