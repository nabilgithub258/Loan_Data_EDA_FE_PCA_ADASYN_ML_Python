
---

# Loan Default Prediction

This project aims to predict whether a borrower will repay their loan or default based on a dataset of approximately 10,000 data points. Given the small and highly imbalanced nature of the dataset, we explored various resampling techniques to improve model performance.

## Project Overview

- **Objective**: Predict loan repayment status.
- **Dataset**: Contains around 10,000 records with borrower information and loan status.
- **Challenge**: The dataset is small and imbalanced, making it difficult to train accurate models.

## Methodology

### Data Preprocessing

1. **Feature Engineering**:
    - Categorical features were one-hot encoded.
    - Numerical features were standardized.

2. **Handling Imbalance**:
    - Various resampling techniques were employed to address class imbalance:
        - **SMOTE (Synthetic Minority Over-sampling Technique)**
        - **SMOTEENN (Combination of SMOTE and Edited Nearest Neighbors)**
        - **ADASYN (Adaptive Synthetic Sampling)**

### Model Building

We used an ensemble approach with the following classifiers:
- **XGBClassifier (Extreme Gradient Boosting)**
- **Logistic Regression**
- **Random Forest Classifier**

These classifiers were combined using a Voting Classifier to leverage the strengths of each model.

### Hyperparameter Tuning

Hyperparameters were optimized using `RandomizedSearchCV` to find the best model configuration.

## Results

Among the resampling techniques used, ADASYN provided the best results, improving the model's ability to handle class imbalance and achieve better overall performance.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- Jupyter Notebook


## Conclusion

This project demonstrates the importance of addressing class imbalance in predictive modeling. By employing advanced resampling techniques like ADASYN, we achieved better performance in predicting loan default status.

---
