"""
Predicting Late Deliveries Using Public E-commerce Order Fulfillment Data
============================================================================

End-to-end pipeline: EDA -> cleaning & feature engineering -> model
training/evaluation -> hyperparameter tuning -> explainability (SHAP).

Refactored from a Jupyter notebook into a single, importable, well-organized
script. Each notebook "part" is now a self-contained function so the
pipeline can be run in full (`python late_delivery_prediction.py`) or
imported and called stage-by-stage from another script / notebook.

Usage
-----
    python late_delivery_prediction.py --data data/DataCoSupplyChainDataset.csv

Directory layout produced
--------------------------
    output/
        clean_stage1.csv
        prepared_dataset.csv
        Model_Comparison.csv
        Final_Model_Comparison.xlsx
        X_train_smote.npy, X_test_scaled.npy, y_train_smote.npy, y_test.npy
        feature_names.csv
    models/
        <model_name>.pkl
        final_model.pkl
        standard_scaler.pkl
        feature_names.csv
    figures/
        *.png  (every plot the notebook produced, saved instead of shown)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:  # SHAP is optional at runtime
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # headless / script-safe backend
plt.style.use("ggplot")
sns.set(font_scale=1.0)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Central configuration for paths and pipeline parameters."""

    data_path: str = "data/DataCoSupplyChainDataset.csv"
    output_dir: Path = Path("output")
    models_dir: Path = Path("models")
    figures_dir: Path = Path("figures")

    missing_value_drop_threshold: float = 40.0  # % missing -> drop column
    test_size: float = 0.30
    random_state: int = 42

    save_figures: bool = True
    run_shap: bool = True

    def __post_init__(self) -> None:
        for path in (self.output_dir, self.models_dir, self.figures_dir):
            path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Small plotting helper (keeps every notebook chart, but saves instead of
# blocking on plt.show())
# ---------------------------------------------------------------------------

def _save_or_show(fig_name: str, cfg: Config) -> None:
    if cfg.save_figures:
        path = cfg.figures_dir / f"{fig_name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure: %s", path)
    plt.close()


# ---------------------------------------------------------------------------
# PART 1 — Data loading & exploratory data analysis
# ---------------------------------------------------------------------------

def load_data(cfg: Config) -> pd.DataFrame:
    """Load the raw DataCo supply chain dataset."""
    logger.info("Loading dataset from %s", cfg.data_path)
    df = pd.read_csv(cfg.data_path, encoding="latin1")
    logger.info("Loaded dataset: %d rows x %d columns", *df.shape)
    return df


def explore_data(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Run the exploratory data analysis performed in Part 1 of the notebook.

    Prints summary statistics and saves every EDA chart to `cfg.figures_dir`.
    Returns the de-duplicated dataframe.
    """
    logger.info("Rows: %d | Columns: %d", *df.shape)
    logger.info("Columns: %s", list(df.columns))
    logger.info("\n%s", df.dtypes)
    logger.info("\n%s", df.describe().T)
    logger.info("\n%s", df.describe(include="object").T)

    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    logger.info("Missing values:\n%s", missing)

    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_percent = missing_percent.sort_values(ascending=False)
    logger.info("Missing value %%:\n%s", missing_percent[missing_percent > 0])

    # Missing value heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Values Heatmap")
    _save_or_show("missing_values_heatmap", cfg)

    # Missing value bar chart
    if len(missing) > 0:
        plt.figure(figsize=(10, 6))
        missing.sort_values().plot(kind="barh")
        plt.title("Missing Values by Feature")
        plt.xlabel("Count")
        _save_or_show("missing_values_barchart", cfg)

    duplicates = df.duplicated().sum()
    logger.info("Duplicate records: %d", duplicates)
    df = df.drop_duplicates()
    logger.info("Shape after de-duplication: %s", df.shape)

    logger.info("Unique values per column:\n%s", df.nunique().sort_values())

    # --- Target variable exploration -------------------------------------
    logger.info("Late_delivery_risk counts:\n%s", df["Late_delivery_risk"].value_counts())

    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x="Late_delivery_risk")
    plt.title("Late Delivery Risk Distribution")
    plt.xlabel("Late Delivery")
    plt.ylabel("Count")
    _save_or_show("target_distribution_count", cfg)

    target_pct = df["Late_delivery_risk"].value_counts(normalize=True) * 100
    logger.info("Target percentage distribution:\n%s", target_pct)

    plt.figure(figsize=(6, 6))
    df["Late_delivery_risk"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90)
    plt.ylabel("")
    plt.title("Late Delivery Distribution")
    _save_or_show("target_distribution_pie", cfg)

    # --- Feature distributions --------------------------------------------
    numerical_features = df.select_dtypes(include=np.number).columns
    categorical_features = df.select_dtypes(include="object").columns
    logger.info("Numerical features: %s", list(numerical_features))
    logger.info("Categorical features: %s", list(categorical_features))

    df[numerical_features].hist(figsize=(20, 18), bins=30)
    _save_or_show("numerical_feature_histograms", cfg)

    plt.figure(figsize=(18, 12))
    corr = df[numerical_features].corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Matrix")
    _save_or_show("correlation_matrix", cfg)

    corr_pairs = corr.unstack().sort_values(kind="quicksort")
    logger.info("Top correlated variable pairs:\n%s", corr_pairs.tail(30))

    plt.figure(figsize=(8, 5))
    sns.countplot(y="Shipping Mode", data=df, order=df["Shipping Mode"].value_counts().index)
    plt.title("Shipping Mode Distribution")
    _save_or_show("shipping_mode_distribution", cfg)

    plt.figure(figsize=(10, 6))
    sns.countplot(y="Delivery Status", data=df, order=df["Delivery Status"].value_counts().index)
    plt.title("Delivery Status")
    _save_or_show("delivery_status_distribution", cfg)

    plt.figure(figsize=(12, 8))
    sns.countplot(
        y="Category Name",
        data=df,
        order=df["Category Name"].value_counts().head(15).index,
    )
    plt.title("Top Product Categories")
    _save_or_show("top_product_categories", cfg)

    plt.figure(figsize=(12, 8))
    sns.countplot(
        y="Order Country",
        data=df,
        order=df["Order Country"].value_counts().head(20).index,
    )
    plt.title("Top Countries")
    _save_or_show("top_countries", cfg)

    plt.figure(figsize=(10, 5))
    sns.histplot(df["Sales"], bins=50, kde=True)
    plt.title("Sales Distribution")
    _save_or_show("sales_distribution", cfg)

    plt.figure(figsize=(10, 5))
    sns.histplot(df["Order Profit Per Order"], bins=50, kde=True)
    plt.title("Order Profit Distribution")
    _save_or_show("order_profit_distribution", cfg)

    sample = df.sample(min(1000, len(df)), random_state=cfg.random_state)
    sns.pairplot(
        sample[
            [
                "Sales",
                "Benefit per order",
                "Order Item Quantity",
                "Order Item Total",
                "Late_delivery_risk",
            ]
        ],
        hue="Late_delivery_risk",
    )
    _save_or_show("pairplot", cfg)

    stage1_path = cfg.output_dir / "clean_stage1.csv"
    df.to_csv(stage1_path, index=False)
    logger.info("Stage 1 (post-EDA) dataset saved to %s", stage1_path)

    return df


# ---------------------------------------------------------------------------
# PART 2 — Cleaning, feature engineering & preparation
# ---------------------------------------------------------------------------

def clean_missing_values(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Drop high-missingness columns and impute the rest."""
    missing_percent = df.isnull().mean() * 100
    drop_cols = missing_percent[missing_percent > cfg.missing_value_drop_threshold].index
    logger.info("Dropping columns with >%.0f%% missing: %s", cfg.missing_value_drop_threshold, list(drop_cols))
    df = df.drop(columns=drop_cols)

    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    assert df.isnull().sum().sum() == 0, "Missing values remain after imputation"
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create date-derived, monetary, and frequency-encoded features."""
    df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"])
    df["shipping date (DateOrders)"] = pd.to_datetime(df["shipping date (DateOrders)"])

    df["Shipping_Duration"] = (
        df["shipping date (DateOrders)"] - df["order date (DateOrders)"]
    ).dt.days
    df["Order_Month"] = df["order date (DateOrders)"].dt.month
    df["Order_Year"] = df["order date (DateOrders)"].dt.year
    df["Order_Day"] = df["order date (DateOrders)"].dt.day
    df["Order_DayOfWeek"] = df["order date (DateOrders)"].dt.day_name()
    df["Weekend_Order"] = np.where(df["Order_DayOfWeek"].isin(["Saturday", "Sunday"]), 1, 0)

    df["Order_Value"] = df["Order Item Quantity"] * df["Order Item Product Price"]
    df["Profit_Margin"] = df["Benefit per order"] / df["Sales"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    df["Customer_Frequency"] = df["Customer Id"].map(df.groupby("Customer Id").size())
    df["Product_Frequency"] = df["Product Card Id"].map(df.groupby("Product Card Id").size())
    df["Market_Frequency"] = df["Market"].map(df.groupby("Market").size())

    # Target: actual shipping time exceeded the scheduled time.
    df["Late_Delivery"] = np.where(
        df["Days for shipment (scheduled)"] < df["Days for shipping (real)"], 1, 0
    )
    logger.info("Target distribution:\n%s", df["Late_Delivery"].value_counts())

    # Drop columns that would leak the target.
    leakage_cols = [
        "Late_delivery_risk",
        "Delivery Status",
        "shipping date (DateOrders)",
        "Days for shipping (real)",
    ]
    df = df.drop(columns=leakage_cols, errors="ignore")

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode binary columns and one-hot encode remaining categoricals."""
    binary_cols = ["Weekend_Order"]
    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    categorical_columns = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df


def prepare_train_test(
    df: pd.DataFrame, cfg: Config
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, List[str]]:
    """Split, scale, and SMOTE-balance the dataset; persist artifacts."""
    X = df.drop("Late_Delivery", axis=1)
    y = df["Late_Delivery"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    X_train_num = X_train[numeric_cols]
    X_test_num = X_test[numeric_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    logger.info("Class distribution before SMOTE:\n%s", y_train.value_counts())

    smote = SMOTE(random_state=cfg.random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    logger.info("Class distribution after SMOTE:\n%s", pd.Series(y_train_smote).value_counts())

    plt.figure(figsize=(6, 5))
    pd.Series(y_train_smote).value_counts().plot(kind="bar")
    plt.title("Training Set After SMOTE")
    _save_or_show("smote_class_distribution", cfg)

    # Persist prepared dataset & supporting objects.
    prepared = pd.concat(
        [pd.DataFrame(X_train_smote, columns=numeric_cols), pd.Series(y_train_smote, name="Late_Delivery")],
        axis=1,
    )
    prepared.to_csv(cfg.output_dir / "prepared_dataset.csv", index=False)

    joblib.dump(scaler, cfg.models_dir / "standard_scaler.pkl")

    np.save(cfg.output_dir / "X_train_smote.npy", X_train_smote)
    np.save(cfg.output_dir / "X_test_scaled.npy", X_test_scaled)
    np.save(cfg.output_dir / "y_train_smote.npy", y_train_smote)
    np.save(cfg.output_dir / "y_test.npy", y_test.to_numpy())

    feature_names = list(numeric_cols)
    pd.DataFrame(feature_names, columns=["Feature"]).to_csv(
        cfg.output_dir / "feature_names.csv", index=False
    )

    logger.info("Training shape: %s | Testing shape: %s", X_train_smote.shape, X_test_scaled.shape)

    return X_train_smote, X_test_scaled, y_train_smote, y_test, feature_names


# ---------------------------------------------------------------------------
# PART 3 — Model training & evaluation
# ---------------------------------------------------------------------------

def build_model_zoo(random_state: int) -> Dict[str, object]:
    """Return the dictionary of candidate models to benchmark."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=random_state),
    }


def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
) -> pd.DataFrame:
    """Train every candidate model, evaluate it, plot diagnostics, and persist it."""
    models = build_model_zoo(cfg.random_state)
    results: List[List] = []

    for name, model in models.items():
        logger.info("=" * 70)
        logger.info("Training: %s", name)
        logger.info("=" * 70)

        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction)
        recall = recall_score(y_test, prediction)
        f1 = f1_score(y_test, prediction)
        roc = roc_auc_score(y_test, probability)
        results.append([name, accuracy, precision, recall, f1, roc])

        logger.info("\n%s", classification_report(y_test, prediction))

        # Confusion matrix
        cm = confusion_matrix(y_test, prediction)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(name)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        _save_or_show(f"confusion_matrix_{name.replace(' ', '_')}", cfg)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, probability)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=name)
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(name)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        _save_or_show(f"roc_curve_{name.replace(' ', '_')}", cfg)

        # Precision-recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, probability)
        plt.figure(figsize=(6, 5))
        plt.plot(recall_curve, precision_curve)
        plt.title(name)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        _save_or_show(f"pr_curve_{name.replace(' ', '_')}", cfg)

        # Cross-validation on the same (train) data used to fit the model.
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        logger.info("Cross-validation accuracy: %s | Mean: %.4f", cv_scores, cv_scores.mean())

        joblib.dump(model, cfg.models_dir / f"{name.replace(' ', '_')}.pkl")

    comparison = pd.DataFrame(
        results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ).sort_values(by="ROC-AUC", ascending=False)

    comparison.to_csv(cfg.output_dir / "Model_Comparison.csv", index=False)
    comparison.to_excel(cfg.output_dir / "Final_Model_Comparison.xlsx", index=False)

    for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=comparison, x=metric, y="Model")
        plt.title(f"{metric} Comparison")
        _save_or_show(f"comparison_{metric.replace(' ', '_').lower()}", cfg)

    comparison.set_index("Model").plot(kind="bar", figsize=(12, 6))
    plt.title("Overall Model Performance")
    plt.ylabel("Score")
    _save_or_show("comparison_overall", cfg)

    return comparison


def save_best_model(comparison: pd.DataFrame, cfg: Config) -> str:
    """Copy the top-ranked model (by ROC-AUC) to models/final_model.pkl."""
    best = comparison.iloc[0]
    logger.info("Best model: %s\n%s", best["Model"], best)

    best_model_path = cfg.models_dir / f"{best['Model'].replace(' ', '_')}.pkl"
    best_model = joblib.load(best_model_path)
    joblib.dump(best_model, cfg.models_dir / "final_model.pkl")
    logger.info("Best model (%s) saved to %s", best["Model"], cfg.models_dir / "final_model.pkl")

    for src in (cfg.models_dir / "standard_scaler.pkl", cfg.output_dir / "feature_names.csv"):
        if src.exists():
            dst = cfg.models_dir / src.name
            if src != dst:
                shutil.copy(src, dst)
                logger.info("Copied %s to %s", src, dst)

    return best["Model"]


# ---------------------------------------------------------------------------
# PART 4 — Hyperparameter tuning & explainability
# ---------------------------------------------------------------------------

def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray, cfg: Config) -> RandomForestClassifier:
    """Grid-search Random Forest hyperparameters."""
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=cfg.random_state),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    logger.info("Best Random Forest params: %s (accuracy=%.4f)", grid.best_params_, grid.best_score_)
    return grid.best_estimator_


def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray, cfg: Config) -> XGBClassifier:
    """Grid-search XGBoost hyperparameters."""
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1],
    }
    grid = GridSearchCV(
        estimator=XGBClassifier(random_state=cfg.random_state, eval_metric="logloss"),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    logger.info("Best XGBoost params: %s (accuracy=%.4f)", grid.best_params_, grid.best_score_)
    return grid.best_estimator_


def explain_model(
    model, X_train: np.ndarray, X_test: np.ndarray, feature_names: List[str], cfg: Config
) -> None:
    """Produce SHAP summary plots and permutation importance for a fitted model."""
    if cfg.run_shap and SHAP_AVAILABLE:
        try:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            _save_or_show("shap_summary", cfg)
        except Exception as exc:  # SHAP can fail on some model/data combos
            logger.warning("SHAP explanation skipped: %s", exc)
    elif cfg.run_shap and not SHAP_AVAILABLE:
        logger.warning("shap is not installed; skipping SHAP explanation.")

    y_dummy = None  # permutation_importance needs y; caller passes via closure if needed


def permutation_feature_importance(
    model, X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str], cfg: Config
) -> pd.DataFrame:
    """Compute and plot permutation feature importance."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=cfg.random_state, n_jobs=-1
    )
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": result.importances_mean}
    ).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df.head(20), x="Importance", y="Feature")
    plt.title("Permutation Feature Importance (Top 20)")
    _save_or_show("permutation_importance", cfg)

    importance_df.to_csv(cfg.output_dir / "feature_importance.csv", index=False)
    return importance_df


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(cfg: Config) -> None:
    """Execute the full pipeline end-to-end."""
    df = load_data(cfg)
    df = explore_data(df, cfg)

    df = clean_missing_values(df, cfg)
    df = engineer_features(df)
    df = encode_features(df)

    X_train, X_test, y_train, y_test, feature_names = prepare_train_test(df, cfg)

    comparison = train_and_evaluate_models(X_train, X_test, y_train, y_test, cfg)
    best_model_name = save_best_model(comparison, cfg)

    best_rf = tune_random_forest(X_train, y_train, cfg)
    joblib.dump(best_rf, cfg.models_dir / "Random_Forest_tuned.pkl")

    best_xgb = tune_xgboost(X_train, y_train, cfg)
    joblib.dump(best_xgb, cfg.models_dir / "XGBoost_tuned.pkl")

    # Explainability on the tuned XGBoost model (typically the strongest performer).
    explain_model(best_xgb, X_train, X_test, feature_names, cfg)
    permutation_feature_importance(best_xgb, X_test, y_test, feature_names, cfg)

    logger.info("Pipeline complete. Best baseline model: %s", best_model_name)
    logger.info("Artifacts written to: %s, %s, %s", cfg.output_dir, cfg.models_dir, cfg.figures_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        dest="data_path",
        default="data/DataCoSupplyChainDataset.csv",
        help="Path to the DataCo supply chain CSV file.",
    )
    parser.add_argument("--output-dir", default="output", help="Directory for generated CSV/NPY artifacts.")
    parser.add_argument("--models-dir", default="models", help="Directory for saved model files.")
    parser.add_argument("--figures-dir", default="figures", help="Directory for saved plots.")
    parser.add_argument("--no-figures", action="store_true", help="Skip saving plots (faster runs).")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP explainability step.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        data_path=args.data_path,
        output_dir=Path(args.output_dir),
        models_dir=Path(args.models_dir),
        figures_dir=Path(args.figures_dir),
        save_figures=not args.no_figures,
        run_shap=not args.no_shap,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
