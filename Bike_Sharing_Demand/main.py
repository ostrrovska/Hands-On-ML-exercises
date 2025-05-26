
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split, cross_val_score


def load_data():
    data, meta = arff.loadarff("dataset_")
    df = pd.DataFrame(data)

    # Decode byte-encoded columns to strings if necessary (common with arff files)
    for col in df.select_dtypes([object]):
        if df[col].dtype.name == "bytes":
            df[col] = df[col].str.decode("utf-8")
    return df

renting = load_data()

#print(renting.head())
#print(renting.describe())
#print(renting.info())

train_set, test_set = train_test_split(renting, test_size=0.2, random_state=42)

X_train = train_set.drop(["count"], axis=1)
y_train = train_set["count"]

X_test = test_set.drop(["count"], axis=1)
y_test = test_set["count"]

from scipy.stats import randint

param_distributions = {
    'n_estimators': randint(100, 400),
    'max_depth': [None] + list(randint(10, 40).rvs(3)),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5)
}

random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions, n_iter=20, cv=5,
                                   scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=42)
std_scaler = StandardScaler()
X_train_scaled = std_scaler.fit_transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

random_search.fit(X_train_scaled, y_train)
best_model = random_search.best_estimator_
predictions = best_model.predict(X_test_scaled)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
forest_rmses = -cross_val_score(best_model, X_train_scaled, y_train, cv=10, scoring="neg_mean_squared_error")
r2 = r2_score(y_test, predictions)

print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", -random_search.best_score_)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
print("Cross Validation RMSEs:", forest_rmses)
print(pd.Series(forest_rmses).describe())

print("predicted vs actual")
print(predictions[:5])
print(y_test[:5].values)
