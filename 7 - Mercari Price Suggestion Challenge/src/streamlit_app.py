from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import _column_transformer as sklearn_column_transformer
import streamlit as st


SRC_DIR = Path(__file__).resolve().parent
MODEL_PATH = SRC_DIR / "mercari_price_model.pkl"


# Compatibility shim for sklearn pickles created in a different runtime.
if not hasattr(sklearn_column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass

    sklearn_column_transformer._RemainderColsList = _RemainderColsList


st.set_page_config(page_title="Mercari Price Predictor", page_icon="🛍️", layout="centered")

st.title("Mercari Price Predictor")
st.caption("Estimate a resale price from product details.")

if not MODEL_PATH.exists():
    st.error("Trained model not found.")
    st.code(f"python {SRC_DIR / 'train_mercari_app_model.py'}")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with st.form("mercari_form"):
    name = st.text_input("Product Name", placeholder="e.g. iPhone 11 128GB")
    brand_name = st.text_input("Brand", placeholder="e.g. Apple")
    category_name = st.text_input("Category", placeholder="e.g. Electronics/Phones/Smartphones")
    item_condition_id = st.selectbox("Item Condition", [1, 2, 3, 4, 5], index=1)
    shipping = st.selectbox(
        "Shipping",
        [0, 1],
        format_func=lambda x: "Seller pays shipping" if x == 1 else "Buyer pays shipping",
    )
    item_description = st.text_area("Description", placeholder="Describe the product")
    submitted = st.form_submit_button("Predict Price")

if submitted:
    row = pd.DataFrame(
        [
            {
                "name": name or "missing",
                "item_condition_id": int(item_condition_id),
                "category_name": category_name or "missing",
                "brand_name": brand_name or "missing",
                "shipping": int(shipping),
                "item_description": item_description or "missing",
            }
        ]
    )

    prediction = float(model.predict(row)[0])

    st.subheader("Estimated Price")
    st.metric("Predicted Price", f"${prediction:,.2f}")

    st.write("Input used for prediction:")
    st.dataframe(row, use_container_width=True)
