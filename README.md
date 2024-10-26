Titanic Survival Prediction Project
This repository contains a machine learning model to predict the survival of Titanic passengers based on demographic and socio-economic features. The model uses a Random Forest Classifier and various preprocessing, feature engineering, and visualization techniques.

Table of Contents
Project Overview
Dataset
Features
Installation
Data Preprocessing
Modeling
Evaluation
Visualization
File Descriptions
Acknowledgments
Project Overview
This project predicts the survival of passengers on the Titanic using demographic, socio-economic, and other features. The model uses a Random Forest Classifier, achieving a validation accuracy of 81.34% when compared to the baseline gender-based submission. The project includes data preprocessing, feature engineering, cross-validation, and visualizations of feature importance and prediction analysis.

Dataset
The dataset used in this project is from the Kaggle Titanic Competition, which includes:

train.csv: Training data with survival labels.
test.csv: Testing data without survival labels.
gender_submission.csv: Baseline gender-based predictions for survival.
Features
Pclass: Passenger class
Sex: Gender
Age: Age of the passenger
SibSp: Number of siblings/spouses aboard
Parch: Number of parents/children aboard
Fare: Ticket fare
FamilySize: Number of family members (SibSp + Parch + 1)
IsAlone: Indicator if the passenger is alone (1 if alone, 0 otherwise)
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Data Preprocessing
Fill Missing Values: Fill missing values in the Age and Fare columns with their median values.
Feature Engineering:
FamilySize is calculated as the sum of SibSp and Parch, plus 1.
IsAlone is created as a binary feature based on FamilySize.
Encoding: Convert Sex to binary (0 for female, 1 for male).
Modeling
Random Forest Classifier: A Random Forest model with 100 trees is used as the primary model.
Standardization: StandardScaler is applied to scale features for optimized performance.
Hyperparameters:
n_estimators=100
random_state=42 for reproducibility
To train the model, run:

python
Copy code
rf_model.fit(X_train_scaled, y_train)
Evaluation
Accuracy: Compared with the baseline gender-based predictions, achieving an accuracy of 81.34%.
Cross-Validation: Achieved a mean cross-validation score of 0.8115.
Classification Report and Confusion Matrix for further insights into model performance.
python
Copy code
print(classification_report(y_train, y_train_pred))
sns.heatmap(confusion_matrix(y_train, y_train_pred), annot=True, cmap='Blues')
Visualization
Survival Rate Analysis:
Survival rate by Pclass, Sex, and Embarked location.
Distribution Plots:
Age and Fare distributions with survival hue.
Feature Importance:
A bar plot showcasing feature importance scores.
python
Copy code
sns.barplot(x='importance', y='feature', data=importances)
plt.title('Feature Importances')
File Descriptions
train.py: Script to preprocess data, train the model, and evaluate it.
submission.csv: Final predictions file for submission.
requirements.txt: List of dependencies.
README.md: Project documentation.
Acknowledgments
Thanks to Kaggle and the open-source community for providing the Titanic dataset and baseline submission.

