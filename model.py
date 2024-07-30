import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

import joblib


class DataProcessor:
    def __init__(self):
        pass

    def split_data(self, df, test_size=0.2, random_state=42):
        # Create target and feature attributes
        target = df["median_house_value"]
        features = df.drop(columns="median_house_value", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
        return X_train, y_train

    def create_pipeline(self, X_train, random_state=42):
        # Create numerical and categorical attributes
        num_attributes = X_train.select_dtypes(include=[np.number]).columns.to_list()
        cat_attributes = X_train.select_dtypes(include=[object]).columns.to_list()

        # Numerical attributes pipeline
        num_pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler()
        )

        # Categorical attributes pipeline
        cat_pipeline = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore")
        )

        # Main preprocessing pipeline
        preprocessing = ColumnTransformer([
            ("numerical_pipeline", num_pipeline, num_attributes),
            ("categorical_pipeline", cat_pipeline, cat_attributes)
        ])
        # Final pipeline
        forest_reg = make_pipeline(
            preprocessing,
            RandomForestRegressor(random_state=random_state)
        )
        return forest_reg


class ModelTrainer:
    def __init__(self):
        pass

    def train_model(self, X_train, y_train, forest_reg, param_grid,
                    cv=3, scoring="neg_root_mean_squared_error"):
        grid_search = GridSearchCV(
            forest_reg,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv
        )
        grid_search.fit(X_train, y_train) # Fit the model
        return grid_search.best_estimator_  # Return best performing model

    def save_best_model(self, best_model, filename):
        joblib.dump(best_model, filename)