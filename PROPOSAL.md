Project Description



This project aims to build a machine learning system capable of predicting product demand for a small retail store and generating simple order recommendations based on the forecasted needs. Using publicly available retail datasets, the objective is to model how much of each product will be sold in the near future, then convert these predictions into reorder suggestions.

The project will be implemented in Python using pandas, NumPy, scikit-learn, and matplotlib. The workflow includes data preprocessing, feature engineering (seasonality, trends, historical lags), model training, evaluation, and automated recommendation logic (for example, “order X units if predicted sales exceed current stock”).

Supervised learning models such as Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor will be trained and compared for forecasting accuracy. This topic offers a clear ML objective, a well-defined target, and datasets large enough for proper validation.



Motivations

I am interested in consulting and in helping businesses integrate AI into their operations. Retail stores regularly face operational decisions related to ordering and inventory planning. This project allows me to work on a practical, business-relevant ML problem that transforms historical sales data into actionable insights.



Expected Challenges and How I’ll Address Them

Seasonality and sales fluctuations: I will engineer time-based features such as moving averages, lag values, and calendar variables to capture recurring patterns.

Uneven sales across products: I will filter for products with sufficient sales history and aggregate data where needed.

Model evaluation: I will use time-aware train/test splits to ensure realistic forecasting and avoid data leakage.



Success Criteria

A cleaned dataset with consistent weekly or daily sales for multiple products.

Three trained and compared ML models (Linear Regression, Random Forest, Gradient Boosting).

Clear evaluation using MAE and RMSE.

A basic order recommendation rule derived from demand predictions.

Visualizations showing past sales, predicted demand, and ordering suggestions.

A documented and structured codebase following the course guidelines.



Stretch Goals

Build a small Streamlit dashboard displaying forecasts and recommendations.

Add uncertainty intervals to predictions.

Incorporate external variables (promotions, holidays) if available.