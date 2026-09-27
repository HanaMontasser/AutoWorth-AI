# Running the AutoWorth AI app locally

1. Put these files in the **same folder**:
   - app.py
   - onehot_encoder.pkl
   - xgboost_model.pkl
   - numerical_cols.pkl
   - categorical_cols.pkl
   - final_columns.pkl
   - requirements.txt

2. (Recommended) Create a virtual environment, then install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   streamlit run app.py
   ```

4. Your browser should open automatically at `http://localhost:8501`.
   If it doesn't, open that address manually.

## Notes
- The app loads the exact encoder and XGBoost model saved from the notebook —
  it does not retrain anything.
- Dropdown options (Make, Model, Transmission, Fuel Type) are pulled directly
  from what the encoder was trained on, so you can't accidentally select a
  category the model has never seen.
- If you see a scikit-learn version warning when loading the .pkl files, it's
  safe as long as predictions still run — but for full reproducibility, use
  the exact versions pinned in requirements.txt.
