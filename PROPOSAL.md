Project Description

This project aims to build a machine learning system capable of predicting product demand for a small retail store and generating simple order recommendations based on forecasted needs. Using publicly available retail datasets, the goal is to model how much of each product will be sold in the near future, then translate these predictions into reorder suggestions.
The project will be implemented in Python, using pandas, NumPy, scikit-learn, and matplotlib. The workflow includes data preprocessing, feature engineering (seasonality, trend, past sales), model training, evaluation, and the generation of an automated order recommendation rule (e.g., “order X units if predicted stock-out risk exceeds a threshold”).
Machine learning models such as Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor will be trained and compared on their ability to forecast product demand. The project targets a clear and measurable problem, backed by a dataset large enough to apply supervised learning and proper validation.

⸻

Motivations

I have always been interested in consulting, especially in helping companies use AI to improve their operations. Retail businesses constantly deal with questions about what to order and when, and this project allows me to explore a concrete, real-world problem in that direction. It connects business thinking with machine learning by transforming sales data into actionable recommendations.

⸻

Expected Challenges and How I’ll Address Them
	•	Seasonality and variability of sales: I will engineer features such as moving averages, lag variables, and day-of-week or month indicators to help models capture recurring patterns.
	•	Uneven product frequency: Some items may have sparse sales; I will handle this through aggregation, filtering, and by selecting products with enough historical data.
	•	Model evaluation: I will use time-aware train/test splits instead of random splits to avoid leakage and ensure realistic forecasting performance.

⸻

Success Criteria
	•	A cleaned dataset with consistent daily or weekly sales information for several products.
	•	At least three trained and compared ML models (Linear Regression, Random Forest, Gradient Boosting).
	•	Clear evaluation using metrics such as MAE and RMSE.
	•	An order recommendation rule based on predicted demand and current inventory assumptions.
	•	Visualizations showing past sales, predicted demand, and recommended orders.
	•	A structured and documented codebase that follows the project’s technical requirements.

⸻

Stretch Goals
	•	Implement a simple Streamlit dashboard to display predictions and recommendations.
	•	Add confidence intervals or uncertainty estimates to predictions.
	•	Extend the model to include promotions, holidays, or weather data if available.