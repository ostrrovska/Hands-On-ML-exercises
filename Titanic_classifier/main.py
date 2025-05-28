import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv('titanic.csv')
    return df

def fill_missing_age_values(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)
    return df

def fill_missing_embarked_values(df):
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    return df

def preprocess_data(df):
    # Keep only significant features
    features_to_keep = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
    df = df[features_to_keep]
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    
    return df

# Load and preprocess data
titanic_data = load_data()
titanic_data = fill_missing_age_values(titanic_data)
titanic_data = fill_missing_embarked_values(titanic_data)
titanic_data = preprocess_data(titanic_data)

# Split into training and test sets
train_set, test_set = train_test_split(titanic_data, test_size=0.2, random_state=42, stratify=titanic_data['Survived'])

X_train = train_set.drop('Survived', axis=1)
y_train = train_set['Survived']

X_test = test_set.drop('Survived', axis=1)
y_test = test_set['Survived']

# Train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Print performance metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title('Confusion Matrix')
plt.show()

# Plot feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





