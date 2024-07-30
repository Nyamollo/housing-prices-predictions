import joblib
import pandas as pd
from data import DataLoader


class ModelDeployer:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, new_data):
        # Load and wrangle new data
        data_loader = DataLoader()
        new_data_wrangled = data_loader.wrangle(new_data)
        # Make predictions
        predictions = self.model.predict(new_data_wrangled)
        return pd.Series(predictions)


if __name__ == "__main__":
    model_path = "housing_price.pkl"
    deployer = ModelDeployer(model_path)
    new_data = pd.read_csv("new_data.csv")
    predictions = deployer.predict(new_data)
    print(predictions)
