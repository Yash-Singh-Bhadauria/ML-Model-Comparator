
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



file_name=input('Give full name of csv file from which you want data:')
# Load the dataset
df = pd.read_csv(file_name)

# # 1. Check for null values
# print("Null values in dataset:")
# print(df.isnull().sum())

# 2. Drop duplicate rows
df = df.drop_duplicates()

# 3. Encode categorical variables
categorical_cols = df.select_dtypes(include='object').columns.drop('class')
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Encode target column
le = LabelEncoder()
df_encoded['class'] = le.fit_transform(df_encoded['class'])  # good=1, bad=0

# 4. Define X and y
X = df_encoded.drop('class', axis=1)
y = df_encoded['class']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
    "SVM (Linear Kernel)": SVC(kernel='linear'),
    "SVM (RBF Kernel)": SVC(kernel='rbf')
}

# 7. Training and Evaluation
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
