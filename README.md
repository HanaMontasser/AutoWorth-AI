🚗 AutoWorth AI — Used Car Price & Deal Advisor

An end-to-end Machine Learning project that predicts the fair market price of a used UK car and compares it against a seller's asking price to produce a Deal Rating (Great Deal → Overpriced).

Problem Statement

Used car buyers often don't know whether an asking price is fair. This project builds a regression model that predicts a car's expected market price from its specifications (make, model, year, mileage, transmission, fuel type, engine size, MPG, tax), then compares that prediction to the seller's price to rate the deal.

Dataset

100,000 UK Used Car Dataset (Kaggle) — listings from 9 manufacturers: Audi, BMW, Ford, Hyundai, Mercedes, Skoda, Toyota, Vauxhall, Volkswagen.

Target: price (continuous → regression problem)
Raw features: model, year, transmission, mileage, fuelType, tax, mpg, engineSize, Make
Two supplementary files (cclass.csv, focus.csv) were partially merged in: only the ~700 genuinely new listings not already present in the main Mercedes/Ford files were kept, to avoid duplicating records.
Data Cleaning
Removed 3 rows with historically impossible year/model combinations (e.g. a "Zafira" dated 1970 — the model wasn't launched until 1999)
Treated engineSize/mpg/tax anomalies (e.g. engineSize = 0 for non-electric cars) as disguised missing values, imputed via group-wise medians (Make + model, falling back to broader groups when a model had no surviving reference rows)
Discovered and relabeled 32 "A Class" listings that were actually AMG performance variants (identified via an anomalous 4.0L engine size)
Removed 1,478 exact-duplicate rows
Investigated price/mileage outliers using IQR + domain knowledge; retained them, since they corresponded to genuine luxury/SUV models (E Class, X5, Q7, etc.), not data errors

Final cleaned dataset: 98,402 rows.

Exploratory Data Analysis

7 visualizations were produced, including price distribution, price vs. mileage/year, average price by manufacturer, price by transmission/fuel type, and a correlation heatmap. Key findings:

Price is right-skewed; luxury models form a long tail
engineSize (r = 0.65) and year (r = 0.50) are the strongest numeric correlates of price
year and mileage are strongly negatively correlated (r = -0.74)
Feature Engineering
Feature	Formula	Rationale
Car_Age	2020 − year	2020 is the latest model year in the dataset, approximating collection time
Mileage_Per_Year	mileage / max(Car_Age, 1)	Captures usage intensity independent of raw age
High_Performance	1 if engineSize ≥ 3.0 else 0	Flags a distinct segment: avg. price £35,528 vs. £15,537 for others
Encoding

Three strategies were compared using a fixed Linear Regression baseline:

Method	Validation R²	Columns
One-Hot Encoding (chosen)	0.8792	219
Target Encoding (K-Fold safe)	0.8585	12
Frequency Encoding	0.7481	12

One-Hot won despite its high dimensionality — preserving each category as its own column captured more signal than compressing it into a single frequency or average-price number.

Models & Evaluation

Five regression models were trained and compared:

Model	Train R²	Val R²	Gap	Val MAE
Random Forest	0.9933	0.9616	0.0317	£1,174.05
XGBoost	0.9640	0.9589	0.0051	£1,317.47
Decision Tree	0.9996	0.9341	0.0655	£1,474.83
Gradient Boosting	0.9043	0.9038	0.0005	£2,103.49
Linear Regression	0.8799	0.8792	0.0007	£2,214.18

Random Forest was tuned via RandomizedSearchCV; it slightly reduced the overfitting gap (0.0317 → 0.0240) at a negligible accuracy cost.

Final Model: XGBoost

XGBoost was selected — not for the single highest validation R², but for its much smaller train-validation gap, indicating stronger generalization.

Final test set evaluation (evaluated once, after model selection):

R²   = 0.9607
MAE  = £1,312.91
RMSE = £1,983.47

This closely matches validation performance, supporting that no data leakage occurred during preprocessing.

Feature Importance

transmission_Manual (0.217) and engineSize (0.058) were the top features. Notably, model_A Class AMG ranked 6th — direct validation of the earlier data-cleaning fix. All three engineered features showed near-zero importance: XGBoost derived the same signal directly from the raw columns they were built from (year, engineSize).

Error Analysis

The largest prediction errors cluster around premium brands (Audi, Mercedes), particularly very old examples of otherwise premium models (e.g. a 1999/2002 Mercedes S Class) — likely due to limited training examples in that combination of age and brand.

Smart Deal Advisor

Compares the predicted price to a seller's asking price and classifies the gap into 5 categories:

Rating	Condition
🟢 Great Deal	> 10% cheaper than predicted
🟢 Good Deal	5–10% cheaper
🟡 Fair Price	within ±5%
🟠 Slightly Overpriced	5–10% more expensive
🔴 Overpriced	> 10% more expensive
Streamlit Application

A Streamlit app (app/app.py) loads the exact trained encoder and XGBoost model (no retraining) and lets a user enter car details and a seller price to get a market price estimate and Deal Rating.

Project Structure
AutoWorth-AI/
├── notebook/
│   └── AutoWorth_AI.ipynb
├── models/
│   ├── onehot_encoder.pkl
│   ├── xgboost_model.json
│   ├── numerical_cols.pkl
│   ├── categorical_cols.pkl
│   └── final_columns.pkl
├── app/
│   └── app.py
├── requirements.txt
└── README.md
How to Run

Notebook: open notebook/AutoWorth_AI.ipynb in Google Colab; run all cells in order.

Streamlit app:

bash
pip install -r requirements.txt
cd app
streamlit run app.py

(the .pkl/.json files from models/ must be in the same folder as app.py)

Technologies Used

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, Streamlit

Conclusions

The final XGBoost model explains ~96% of price variance on unseen test data (MAE ≈ £1,313), with the largest remaining errors concentrated in premium brands underrepresented in the training data. The project prioritized evidence-based decisions throughout — cleaning choices, encoding selection, and final model choice were all validated with data rather than assumed.
