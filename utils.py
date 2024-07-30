from data import DataLoader
from model import DataProcessor, ModelTrainer


def main():
    # Get the data
    data_loader = DataLoader()
    df = data_loader.download_data()
    wrangled_df = data_loader.wrangle(df)

    # Process data and create pipeline
    data_processor = DataProcessor()
    X_train, y_train = data_processor.split_data(wrangled_df)
    forest_reg = data_processor.create_pipeline(X_train)

    # Train the model
    param_grid = {
        "columntransformer__numerical_pipeline__simpleimputer__strategy": ["median", "mean"],
        "randomforestregressor__max_depth": [None, 15, 30, 45],
        "randomforestregressor__n_estimators": range(25, 100, 25)
        }

    model_trainer = ModelTrainer()
    best_model = model_trainer.train_model(X_train, y_train, forest_reg, param_grid)

    # Save model
    filename = "housing_price.pkl"
    model_trainer.save_best_model(best_model, filename)

if __name__ == "__main__":
    main()

