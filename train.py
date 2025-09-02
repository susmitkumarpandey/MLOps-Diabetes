import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Read the dataset
df=pd.read_csv('diabetes.csv')

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# Impute missing values with mean
imputer = SimpleImputer(strategy='median')
df[cols_with_zero] = imputer.fit_transform(df[cols_with_zero])

#check if data is imbalanced
x= df.drop('Outcome', axis=1)
y=df['Outcome']

# 4.Apply SMOTE for handling class imbalance
smote = SMOTE()
transform_feature, transform_label = smote.fit_resample(x,y) 

# 4.split data into train and test
x_train,  x_test, y_train, y_test = train_test_split(transform_feature, transform_label, test_size=0.2, random_state=42, stratify=transform_label)

# 6.Feature Scaling
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

# 7.Logistic Regression Model
model = LogisticRegression()
model.fit(x_train_scaler, y_train)

# 8.Predictions
y_pred = model.predict(x_test_scaler)

# 9.Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 10.Saving the model
with open('app/diabetes_model.pkl','wb') as f:
    pickle.dump((scaler, model), f)
print("Model saved successfully")