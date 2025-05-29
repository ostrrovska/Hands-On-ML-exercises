import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def load_data():
    return pd.read_csv('combined_data.csv/combined_data.csv')

email_data = load_data()

print(email_data.head())
print(email_data.info())
print(email_data.describe())

pipeline = make_pipeline(
    CountVectorizer(binary=True)
)

X = email_data['text']
y = email_data['label']

X_transformed = pipeline.fit_transform(X)

train_X, test_X, train_y, test_y = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(train_X, train_y)

y_pred = sgd_clf.predict(test_X)

print("cross validation score: ", cross_val_score(sgd_clf, X_transformed, y, cv=3, scoring='accuracy'))
print("dummy classifier score: ", cross_val_score(DummyClassifier(), X_transformed, y, cv=3, scoring='accuracy'))
print("confusion matrix: ", confusion_matrix(test_y, y_pred))
print("classification report: ", classification_report(test_y, y_pred))




