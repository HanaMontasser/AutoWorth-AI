"""
AutoWorth AI — Used Car Price & Deal Advisor
Streamlit application.

This app loads the SAME preprocessing (OneHotEncoder) and the SAME trained
XGBoost model produced in the AutoWorth AI notebook, so predictions here match
the model's validated performance. No retraining or re-fitting happens here.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoWorth AI",
    page_icon="🚗",
    layout="centered",
)

REFERENCE_YEAR = 2020  # must match the reference year used during training


# ----------------------------------------------------------------------------
# Load the saved pipeline pieces (cached so this only runs once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    encoder = joblib.load("onehot_encoder.pkl")

    # loaded via XGBoost's own portable JSON format (not joblib/pickle) —
    # this avoids "input stream corrupted" errors that can happen when a
    # pickled XGBoost model is moved between different machines/OSes
    model = XGBRegressor()
    model.load_model("xgboost_model.json")

    numerical_cols = joblib.load("numerical_cols.pkl")
    categorical_cols = joblib.load("categorical_cols.pkl")
    final_columns = joblib.load("final_columns.pkl")
    return encoder, model, numerical_cols, categorical_cols, final_columns


encoder, model, numerical_cols, categorical_cols, final_columns = load_pipeline()

# dropdown options come directly from what the encoder was trained on —
# this guarantees the user can never select a category the model has never seen
make_options = sorted(encoder.categories_[categorical_cols.index("Make")])
model_options = sorted(encoder.categories_[categorical_cols.index("model")])
transmission_options = sorted(encoder.categories_[categorical_cols.index("transmission")])
fueltype_options = sorted(encoder.categories_[categorical_cols.index("fuelType")])


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🚗 AutoWorth AI")
st.caption("Used Car Price & Deal Advisor")

st.divider()

# ----------------------------------------------------------------------------
# Section 1: Car Details
# ----------------------------------------------------------------------------
st.subheader("Car Details")

col1, col2 = st.columns(2)

with col1:
    make = st.selectbox("Make", make_options)
    year = st.number_input("Year", min_value=1995, max_value=REFERENCE_YEAR, value=2018, step=1)
    transmission = st.selectbox("Transmission", transmission_options)
    engine_size = st.number_input("Engine Size (L)", min_value=0.0, max_value=6.6, value=1.6, step=0.1)
    tax = st.number_input("Annual Tax (£)", min_value=0, max_value=580, value=145, step=5)

with col2:
    model_name = st.selectbox("Model", model_options)
    mileage = st.number_input("Mileage", min_value=0, max_value=323000, value=30000, step=500)
    fuel_type = st.selectbox("Fuel Type", fueltype_options)
    mpg = st.number_input("MPG", min_value=0.0, max_value=470.8, value=55.0, step=1.0)

st.divider()

# ----------------------------------------------------------------------------
# Section 2: Deal Check
# ----------------------------------------------------------------------------
st.subheader("🚦 Deal Check")
seller_price = st.number_input("Seller's Asking Price (£)", min_value=0, value=15000, step=100)

predict_clicked = st.button("PREDICT MARKET PRICE", type="primary", use_container_width=True)


# ----------------------------------------------------------------------------
# Feature engineering — must mirror the notebook EXACTLY
# ----------------------------------------------------------------------------
def build_feature_row(make, model_name, year, mileage, transmission, fuel_type,
                       engine_size, mpg, tax):
    row = pd.DataFrame([{
        "model": model_name,
        "year": year,
        "transmission": transmission,
        "mileage": mileage,
        "fuelType": fuel_type,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engine_size,
        "Make": make,
    }])

    # same three engineered features created in the notebook, same formulas
    row["Car_Age"] = REFERENCE_YEAR - row["year"]
    row["Mileage_Per_Year"] = row["mileage"] / row["Car_Age"].replace(0, 1)
    row["High_Performance"] = (row["engineSize"] >= 3.0).astype(int)

    return row


def encode_and_align(row, encoder, numerical_cols, categorical_cols, final_columns):
    # transform categorical columns with the SAME fitted encoder used in training
    encoded_array = encoder.transform(row[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=row.index)

    full_row = pd.concat([row[numerical_cols], encoded_df], axis=1)

    # reindex guarantees the exact same column order the model was trained on;
    # this is critical for a tree-based model like XGBoost
    full_row = full_row.reindex(columns=final_columns, fill_value=0)
    return full_row


# ----------------------------------------------------------------------------
# Smart Deal Advisor logic (same categories defined in the notebook)
# ----------------------------------------------------------------------------
def get_deal_rating(estimated_price, seller_price):
    percent_diff = ((seller_price - estimated_price) / estimated_price) * 100

    if percent_diff <= -10:
        return "🟢 GREAT DEAL", "Priced well below estimated market value", percent_diff
    elif percent_diff <= -5:
        return "🟢 GOOD DEAL", "Priced below estimated market value", percent_diff
    elif percent_diff <= 5:
        return "🟡 FAIR PRICE", "Priced close to estimated market value", percent_diff
    elif percent_diff <= 10:
        return "🟠 SLIGHTLY OVERPRICED", "Priced above estimated market value", percent_diff
    else:
        return "🔴 OVERPRICED", "Priced well above estimated market value", percent_diff


# ----------------------------------------------------------------------------
# Run prediction and display results
# ----------------------------------------------------------------------------
if predict_clicked:
    row = build_feature_row(
        make, model_name, year, mileage, transmission,
        fuel_type, engine_size, mpg, tax
    )
    full_row = encode_and_align(row, encoder, numerical_cols, categorical_cols, final_columns)

    estimated_price = float(model.predict(full_row)[0])
    rating, rating_caption, percent_diff = get_deal_rating(estimated_price, seller_price)
    difference = seller_price - estimated_price

    st.divider()
    st.subheader("💰 Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Market Price", f"£{estimated_price:,.0f}")
    c2.metric("Seller Price", f"£{seller_price:,.0f}")
    c3.metric(
        "Difference",
        f"£{abs(difference):,.0f} {'cheaper' if difference < 0 else 'more expensive'}",
        f"{percent_diff:+.1f}%",
    )

    st.markdown(f"### {rating}")
    st.caption(rating_caption)

    with st.expander("⚠️ Model reliability note"):
        st.write(
            "This model is generally accurate (R² ≈ 0.96 on unseen test data), "
            "but is noticeably less reliable for premium/luxury brands (e.g. "
            "Audi, Mercedes) and for very old examples of otherwise premium "
            "models. Treat estimates for those cases with more caution."
        )
