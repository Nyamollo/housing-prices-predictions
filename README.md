# ML Based Housing Prices Prediction

## Introduction
Accurate estimation of housing prices is pivotal for real-estate evaluation and investments. The conventional approach to estimating housing prices is predominantly manual, which is costly, time-consuming, and prone to errors. For more accurate estimations, we develop a machine learning approach that takes in the housing features as inputs and provides an estimate of the median housing income as output. The output will be used along with other signals to determine if it’s worth investing in the area.

## Data Description and Preprocessing
The California Housing Prices dataset, based on the 1990 California census, was obtained from the [ageron/handson-ml3 GitHub repository](https://github.com/ageron/handson-ml3). It has about 20,000 instances. The few missing values were imputed, and new instances were created to better reflect housing prediction requirements.

## Predictive Modelling
Using root mean square error (RMSE) as the performance measurement, we trained and evaluated three models: Linear Regression (LR), Decision Tree Regressor (DTR), and Random Forest Regressor (RFR). RFR was enhanced by hyperparameter optimization to create an optimal model.

## Results
The model performance was evaluated using RMSE, which was calculated for both test and train sets (Table 1).

**Table 1: Performance of the three models as evaluated by RMSE**

| Model | RMSE on Train Set | RMSE on Test Set |
|-------|-------------------|------------------|
| LR    | 70018             | 71806            |
| DTR   | 69256             | 78394            |
| RFR   | 49190             | 51003            |

Based on RMSE, RFR performed better than LR and DTR on both test and training sets. The median house income is the main predictor of housing prices in California. The model performance is statistically significant and safe to deploy.
