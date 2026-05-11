from __future__ import annotations

import os
import pickle
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import py7zr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SRC_DIR = Path(__file__).resolve().parent
DEFAULT_ZIP_PATH = Path("/content/drive/MyDrive/Colab Data Dosyaları/mercari-price-suggestion-challenge.zip")
MODEL_PATH = SRC_DIR / "mercari_price_model.pkl"


def find_zip_member(members: list[str], filename: str) -> str:
    for member in members:
        if member.endswith(filename):
            return member
    raise FileNotFoundError(f"{filename} was not found inside the zip file.")


def read_7z_tsv_from_zip(zip_path: Path, member_name: str, nrows: int | None = None) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmpdir:
        local_7z_path = os.path.join(tmpdir, os.path.basename(member_name))

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            with zip_ref.open(member_name) as src_file, open(local_7z_path, "wb") as dst_file:
                dst_file.write(src_file.read())

        with py7zr.SevenZipFile(local_7z_path, mode="r") as z:
            z.extractall(path=tmpdir)

        tsv_candidates: list[str] = []
        for root, _, files in os.walk(tmpdir):
            for file_name in files:
                if file_name.endswith(".tsv"):
                    tsv_candidates.append(os.path.join(root, file_name))

        if not tsv_candidates:
            raise FileNotFoundError("No TSV file could be extracted from the 7z archive.")

        return pd.read_csv(tsv_candidates[0], sep="\t", nrows=nrows)


def load_training_data(zip_path: Path, sample_rows: int = 200_000) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_members = zip_ref.namelist()

    train_member = find_zip_member(zip_members, "train.tsv.7z")
    train_df = read_7z_tsv_from_zip(zip_path, train_member, nrows=sample_rows)

    train_df["category_name"] = train_df["category_name"].fillna("missing")
    train_df["brand_name"] = train_df["brand_name"].fillna("missing")
    train_df["item_description"] = train_df["item_description"].fillna("missing")
    train_df["name"] = train_df["name"].fillna("missing")
    return train_df


def build_pipeline() -> Pipeline:
    feature_columns = [
        "name",
        "item_condition_id",
        "category_name",
        "brand_name",
        "shipping",
        "item_description",
    ]
    categorical_features = ["name", "category_name", "brand_name", "item_description"]
    numerical_features = ["item_condition_id", "shipping"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numerical_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=18,
                    min_samples_split=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.feature_columns_ = feature_columns
    return model


def main() -> None:
    zip_path = DEFAULT_ZIP_PATH
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip was not found: {zip_path}")

    train_df = load_training_data(zip_path)

    feature_columns = [
        "name",
        "item_condition_id",
        "category_name",
        "brand_name",
        "shipping",
        "item_description",
    ]

    x = train_df[feature_columns].copy()
    y = train_df["price"].copy()

    for col in ["name", "category_name", "brand_name", "item_description"]:
        x[col] = x[col].astype(str)

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

    model = build_pipeline()
    model.fit(x_train, y_train)

    valid_preds = model.predict(x_valid)
    rmse = root_mean_squared_error(y_valid, valid_preds)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Validation RMSE: {rmse:.5f}")


if __name__ == "__main__":
    main()
