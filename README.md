# Adult Income Classification

Predicts whether a person's income is `<=50K` or `>50K` from demographic and employment data, using the Kaggle Adult Income dataset.

## What I did
- Loaded and inspected the data, checked for missing values and duplicates
- Did EDA on income vs demographic/employment features
- Engineered features, then did stratified train/validation/test splits
- Scaled numerical features and encoded categorical ones
- Trained Logistic Regression and Random Forest, compared on validation
- Picked the best model by validation F1, then evaluated once on the held-out test set

## Results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.808 | 0.567 | 0.846 | 0.679 | 0.909 |
| Random Forest | 0.840 | 0.629 | 0.810 | 0.708 | 0.915 |
| Majority Baseline | 0.761 | 0.000 | 0.000 | 0.000 | 0.500 |

Random Forest was selected on validation F1. Final test metrics: accuracy 0.832, precision 0.613, recall 0.815, F1 0.699, ROC-AUC 0.916.

## Run it
1. Download the Adult Income dataset from Kaggle and place `adult.csv` where the notebook expects it
2. `pip install -r requirements.txt`
3. Open `notebooks/classification.ipynb`

## Structure
- `notebooks/` - main notebook
- `results/` - confusion matrix and ROC curve plots
- `models/` - saved model


**Trained model:** [Download the .joblib](https://huggingface.co/RhythmThapa/rhythm-adult-income-random-forest/resolve/main/adult_income_model.joblib?download=true)
