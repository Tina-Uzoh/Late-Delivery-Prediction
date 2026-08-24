"""
Late Delivery Prediction - Streamlit Web App
=============================================
An interactive web application for predicting late deliveries using
supply chain data. Built from the DataCo Supply Chain Dataset pipeline.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
from datetime import datetime
from io import BytesIO

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Late Delivery Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    defaults = {
        'df': None,
        'df_clean': None,
        'df_prepared': None,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'feature_names': None,
        'models': {},
        'comparison': None,
        'best_model': None,
        'scaler': None,
        'trained': False,
        'eda_done': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def load_data(file):
    """Load CSV data with latin1 encoding."""
    df = pd.read_csv(file, encoding="latin1")
    return df


def get_numerical_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_cols(df):
    return df.select_dtypes(include=['object']).columns.tolist()


def create_download_link(df, filename, text):
    """Create a download link for a dataframe."""
    csv = df.to_csv(index=False)
    b64 = csv.encode()
    return f'<a href="data:file/csv;base64,{b64.decode()}" download="{filename}">{text}</a>'


# =============================================================================
# DATA CLEANING & FEATURE ENGINEERING
# =============================================================================

def clean_data(df, missing_threshold=40.0):
    """Clean missing values and drop high-missing columns."""
    df = df.copy()

    # Drop high missing columns
    missing_percent = df.isnull().mean() * 100
    drop_cols = missing_percent[missing_percent > missing_threshold].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Impute remaining
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    return df, drop_cols


def engineer_features(df):
    """Create new features from the dataset."""
    df = df.copy()

    # Date parsing
    date_cols = [c for c in df.columns if 'date' in c.lower() and 'DateOrders' in c]
    if len(date_cols) >= 2:
        order_date_col = [c for c in date_cols if 'order' in c.lower()][0]
        ship_date_col = [c for c in date_cols if 'shipping' in c.lower()][0]

        df[order_date_col] = pd.to_datetime(df[order_date_col], errors='coerce')
        df[ship_date_col] = pd.to_datetime(df[ship_date_col], errors='coerce')

        df["Shipping_Duration"] = (df[ship_date_col] - df[order_date_col]).dt.days
        df["Order_Month"] = df[order_date_col].dt.month
        df["Order_Year"] = df[order_date_col].dt.year
        df["Order_Day"] = df[order_date_col].dt.day
        df["Order_DayOfWeek"] = df[order_date_col].dt.day_name()
        df["Weekend_Order"] = np.where(df["Order_DayOfWeek"].isin(["Saturday", "Sunday"]), 1, 0)

    # Monetary features
    if 'Order Item Quantity' in df.columns and 'Order Item Product Price' in df.columns:
        df["Order_Value"] = df["Order Item Quantity"] * df["Order Item Product Price"]

    if 'Benefit per order' in df.columns and 'Sales' in df.columns:
        df["Profit_Margin"] = df["Benefit per order"] / df["Sales"].replace(0, np.nan)
        df["Profit_Margin"] = df["Profit_Margin"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Frequency features
    if 'Customer Id' in df.columns:
        df["Customer_Frequency"] = df["Customer Id"].map(df.groupby("Customer Id").size())
    if 'Product Card Id' in df.columns:
        df["Product_Frequency"] = df["Product Card Id"].map(df.groupby("Product Card Id").size())
    if 'Market' in df.columns:
        df["Market_Frequency"] = df["Market"].map(df.groupby("Market").size())

    # Target creation
    if 'Days for shipment (scheduled)' in df.columns and 'Days for shipping (real)' in df.columns:
        df["Late_Delivery"] = np.where(
            df["Days for shipment (scheduled)"] < df["Days for shipping (real)"], 1, 0
        )

    # Drop leakage columns
    leakage_cols = [
        "Late_delivery_risk", "Delivery Status",
        "shipping date (DateOrders)", "Days for shipping (real)"
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore")

    return df


def encode_features(df):
    """Encode categorical features."""
    df = df.copy()

    # Label encode binary
    if "Weekend_Order" in df.columns:
        df["Weekend_Order"] = LabelEncoder().fit_transform(df["Weekend_Order"].astype(str))

    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def prepare_data(df, test_size=0.3, random_state=42):
    """Split, scale, and SMOTE balance the data."""
    if "Late_Delivery" not in df.columns:
        raise ValueError("Target column 'Late_Delivery' not found. Ensure feature engineering is done.")

    X = df.drop("Late_Delivery", axis=1)
    y = df["Late_Delivery"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    X_train_num = X_train[numeric_cols]
    X_test_num = X_test[numeric_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    feature_names = list(numeric_cols)

    return X_train_smote, X_test_scaled, y_train_smote, y_test, feature_names, scaler


# =============================================================================
# MODEL TRAINING
# =============================================================================

def build_models(random_state=42):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=random_state),
    }


def train_models(X_train, X_test, y_train, y_test):
    models = build_models()
    results = []
    trained_models = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}... ({i+1}/{len(models)})")

        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction, zero_division=0)
        recall = recall_score(y_test, prediction, zero_division=0)
        f1 = f1_score(y_test, prediction, zero_division=0)
        roc = roc_auc_score(y_test, probability)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc
        })

        trained_models[name] = {
            "model": model,
            "predictions": prediction,
            "probabilities": probability
        }

        progress_bar.progress((i + 1) / len(models))

    status_text.text("Training complete!")
    progress_bar.empty()

    comparison = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    return comparison, trained_models


def tune_best_model(X_train, y_train, model_name, random_state=42):
    """Hyperparameter tuning for the best model."""
    if model_name == "Random Forest":
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }
        model = RandomForestClassifier(random_state=random_state)
    elif model_name == "XGBoost":
        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5, 7],
        }
        model = XGBClassifier(random_state=random_state, eval_metric="logloss")
    else:
        return None

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942026.png", width=80)
    st.title("📦 Late Delivery Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📤 Upload Data", "📊 EDA", "🔧 Preprocessing", "🤖 Model Training", 
         "📈 Results", "🔮 Predict", "ℹ️ About"]
    )

    st.markdown("---")
    st.markdown("**Settings**")
    test_size = st.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
    random_state = st.number_input("Random State", 0, 9999, 42)
    missing_threshold = st.slider("Missing Threshold (%)", 10, 90, 40, 5)

    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")


# =============================================================================
# HOME PAGE
# =============================================================================

if page == "🏠 Home":
    st.markdown('<div class="main-header">Late Delivery Prediction App</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">End-to-end ML pipeline for supply chain delivery risk analysis</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📤 Upload</h3>
            <p>Upload your supply chain CSV dataset</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Train</h3>
            <p>Train & compare multiple ML models</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔮 Predict</h3>
            <p>Make predictions on new orders</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🚀 Quick Start")
    st.markdown("""
    1. **Upload Data** - Load your `DataCoSupplyChainDataset.csv` file
    2. **Explore EDA** - Visualize distributions, correlations, and missing values
    3. **Preprocess** - Clean data, engineer features, and prepare for training
    4. **Train Models** - Compare Logistic Regression, Random Forest, XGBoost, and more
    5. **View Results** - See performance metrics, confusion matrices, and ROC curves
    6. **Predict** - Use the best model to predict late delivery risk
    """)

    if st.session_state.df is not None:
        st.success(f"✅ Dataset loaded: {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} columns")


# =============================================================================
# UPLOAD DATA PAGE
# =============================================================================

elif page == "📤 Upload Data":
    st.header("📤 Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV file (DataCo Supply Chain Dataset)",
        type=["csv"],
        help="Upload the supply chain dataset CSV file"
    )

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.session_state.df = df
            st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

            st.subheader("Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", f"{df.shape[0]:,}")
            col2.metric("Columns", df.shape[1])
            col3.metric("Numeric", len(get_numerical_cols(df)))
            col4.metric("Categorical", len(get_categorical_cols(df)))

            st.subheader("Column Information")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values,
                'Unique': df.nunique().values
            })
            st.dataframe(info_df, use_container_width=True)

            # Show target info if present
            if 'Late_delivery_risk' in df.columns:
                st.info("🎯 Target column 'Late_delivery_risk' detected!")

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    elif st.session_state.df is not None:
        st.info("Using previously loaded dataset.")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)


# =============================================================================
# EDA PAGE
# =============================================================================

elif page == "📊 EDA":
    st.header("📊 Exploratory Data Analysis")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first!")
        st.stop()

    df = st.session_state.df

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Missing Values", "Distributions", "Correlations", "Target Analysis"
    ])

    # --- Overview Tab ---
    with tab1:
        st.subheader("Statistical Summary")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Numerical Features**")
            st.dataframe(df.describe().T, use_container_width=True)
        with col2:
            st.markdown("**Categorical Features**")
            cat_df = df.describe(include="object").T if len(get_categorical_cols(df)) > 0 else pd.DataFrame()
            st.dataframe(cat_df, use_container_width=True)

        st.subheader("Duplicate Records")
        dupes = df.duplicated().sum()
        st.metric("Duplicate Rows", dupes)

    # --- Missing Values Tab ---
    with tab2:
        st.subheader("Missing Values Analysis")

        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) > 0:
            missing_pct = (missing / len(df) * 100).round(2)
            miss_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Missing %': missing_pct.values
            })
            st.dataframe(miss_df, use_container_width=True)

            fig = px.bar(
                miss_df, x='Missing Count', y='Column', orientation='h',
                color='Missing %', color_continuous_scale='Reds',
                title='Missing Values by Column'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")

    # --- Distributions Tab ---
    with tab3:
        st.subheader("Feature Distributions")

        num_cols = get_numerical_cols(df)
        if num_cols:
            selected_col = st.selectbox("Select numerical feature", num_cols)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=selected_col, nbins=50, marginal="box",
                                   title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'Late_delivery_risk' in df.columns:
                    fig = px.histogram(df, x=selected_col, color="Late_delivery_risk",
                                       nbins=50, barmode="overlay",
                                       title=f"{selected_col} by Delivery Risk")
                    st.plotly_chart(fig, use_container_width=True)

        cat_cols = get_categorical_cols(df)
        if cat_cols:
            selected_cat = st.selectbox("Select categorical feature", cat_cols)

            value_counts = df[selected_cat].value_counts().head(20).reset_index()
            value_counts.columns = [selected_cat, 'Count']

            fig = px.bar(value_counts, x='Count', y=selected_cat, orientation='h',
                        title=f"Top 20 values in {selected_cat}")
            st.plotly_chart(fig, use_container_width=True)

    # --- Correlations Tab ---
    with tab4:
        st.subheader("Correlation Matrix")

        num_cols = get_numerical_cols(df)
        if len(num_cols) > 1:
            corr = df[num_cols].corr()

            fig = px.imshow(corr, text_auto='.2f', aspect="auto",
                           color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                           title="Correlation Heatmap")
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)

            # Top correlations
            corr_pairs = corr.unstack().sort_values(kind="quicksort")
            corr_pairs = corr_pairs[corr_pairs < 1]  # Remove self-correlations
            st.subheader("Top Positive Correlations")
            st.dataframe(corr_pairs.tail(10).reset_index().rename(
                columns={0: 'Correlation', 'level_0': 'Feature 1', 'level_1': 'Feature 2'}
            ), use_container_width=True)
        else:
            st.info("Not enough numerical columns for correlation analysis.")

    # --- Target Analysis Tab ---
    with tab5:
        st.subheader("Target Variable Analysis")

        if 'Late_delivery_risk' in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                target_counts = df['Late_delivery_risk'].value_counts()
                fig = px.pie(values=target_counts.values, names=target_counts.index,
                            title="Late Delivery Risk Distribution")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                target_pct = df['Late_delivery_risk'].value_counts(normalize=True) * 100
                st.metric("Late Delivery Rate", f"{target_pct.get(1, 0):.1f}%")
                st.metric("On-Time Delivery Rate", f"{target_pct.get(0, 0):.1f}%")

            if 'Shipping Mode' in df.columns:
                st.subheader("Late Delivery by Shipping Mode")
                crosstab = pd.crosstab(df['Shipping Mode'], df['Late_delivery_risk'],
                                      normalize='index') * 100
                fig = px.bar(crosstab, barmode='group', title="Late Delivery % by Shipping Mode")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Target column 'Late_delivery_risk' not found in dataset.")

    st.session_state.eda_done = True


# =============================================================================
# PREPROCESSING PAGE
# =============================================================================

elif page == "🔧 Preprocessing":
    st.header("🔧 Data Preprocessing & Feature Engineering")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first!")
        st.stop()

    df = st.session_state.df

    st.subheader("Step 1: Data Cleaning")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before Cleaning**")
        st.write(f"Shape: {df.shape}")
        st.write(f"Missing values: {df.isnull().sum().sum()}")

    if st.button("🧹 Clean Data", type="primary"):
        with st.spinner("Cleaning data..."):
            df_clean, dropped_cols = clean_data(df, missing_threshold)
            st.session_state.df_clean = df_clean

            if dropped_cols:
                st.info(f"Dropped columns (> {missing_threshold}% missing): {', '.join(dropped_cols)}")

    if st.session_state.df_clean is not None:
        df_clean = st.session_state.df_clean
        with col2:
            st.markdown("**After Cleaning**")
            st.write(f"Shape: {df_clean.shape}")
            st.write(f"Missing values: {df_clean.isnull().sum().sum()}")
        st.success("✅ Data cleaned successfully!")

    st.markdown("---")
    st.subheader("Step 2: Feature Engineering")

    if st.session_state.df_clean is None:
        st.info("Clean the data first to enable feature engineering.")
        st.stop()

    if st.button("⚙️ Engineer Features", type="primary"):
        with st.spinner("Engineering features..."):
            df_engineered = engineer_features(st.session_state.df_clean)
            st.session_state.df_engineered = df_engineered

            new_features = [c for c in df_engineered.columns if c not in st.session_state.df_clean.columns]
            if new_features:
                st.success(f"✅ Created {len(new_features)} new features: {', '.join(new_features)}")
            else:
                st.info("No new features created (required columns may be missing).")

    if hasattr(st.session_state, 'df_engineered') and st.session_state.df_engineered is not None:
        df_eng = st.session_state.df_engineered

        st.subheader("Engineered Dataset Preview")
        st.dataframe(df_eng.head(10), use_container_width=True)

        if "Late_Delivery" in df_eng.columns:
            st.info(f"🎯 Target distribution: {df_eng['Late_Delivery'].value_counts().to_dict()}")

    st.markdown("---")
    st.subheader("Step 3: Encoding & Train/Test Split")

    if not hasattr(st.session_state, 'df_engineered') or st.session_state.df_engineered is None:
        st.info("Engineer features first to proceed.")
        st.stop()

    if st.button("🔀 Prepare Train/Test Split", type="primary"):
        with st.spinner("Encoding and splitting data..."):
            try:
                df_encoded = encode_features(st.session_state.df_engineered)

                X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(
                    df_encoded, test_size=test_size, random_state=random_state
                )

                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.feature_names = feature_names
                st.session_state.scaler = scaler

                col1, col2, col3 = st.columns(3)
                col1.metric("Training Samples", f"{X_train.shape[0]:,}")
                col2.metric("Testing Samples", f"{X_test.shape[0]:,}")
                col3.metric("Features", X_train.shape[1])

                st.success("✅ Data prepared for training!")

                # Show class distribution
                st.subheader("Class Distribution")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Before SMOTE**")
                    st.bar_chart(y_train.value_counts())
                with col2:
                    st.markdown("**After SMOTE**")
                    st.bar_chart(pd.Series(y_train).value_counts())

            except Exception as e:
                st.error(f"Error during preparation: {str(e)}")


# =============================================================================
# MODEL TRAINING PAGE
# =============================================================================

elif page == "🤖 Model Training":
    st.header("🤖 Model Training")

    if st.session_state.X_train is None:
        st.warning("⚠️ Please complete preprocessing first!")
        st.stop()

    st.subheader("Train Multiple Models")
    st.markdown("Train and compare: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost")

    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            comparison, trained_models = train_models(
                st.session_state.X_train,
                st.session_state.X_test,
                st.session_state.y_train,
                st.session_state.y_test
            )

            st.session_state.comparison = comparison
            st.session_state.models = trained_models
            st.session_state.trained = True

            st.success("✅ All models trained successfully!")

    if st.session_state.trained and st.session_state.comparison is not None:
        st.subheader("Model Comparison")
        st.dataframe(st.session_state.comparison.style.highlight_max(subset=[
            'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'
        ], color='green'), use_container_width=True)

        # Best model
        best_model_name = st.session_state.comparison.iloc[0]['Model']
        st.info(f"🏆 Best Model: **{best_model_name}** (by ROC-AUC)")

        # Hyperparameter tuning
        st.markdown("---")
        st.subheader("Hyperparameter Tuning")

        if best_model_name in ["Random Forest", "XGBoost"]:
            if st.button(f"🔧 Tune {best_model_name}", type="primary"):
                with st.spinner(f"Tuning {best_model_name}..."):
                    tuned_model, best_params, best_score = tune_best_model(
                        st.session_state.X_train,
                        st.session_state.y_train,
                        best_model_name,
                        random_state
                    )

                    st.session_state.best_model = tuned_model

                    st.success(f"✅ Tuning complete! Best score: {best_score:.4f}")
                    st.json(best_params)

                    # Evaluate tuned model
                    tuned_pred = tuned_model.predict(st.session_state.X_test)
                    tuned_prob = tuned_model.predict_proba(st.session_state.X_test)[:, 1]

                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Accuracy", f"{accuracy_score(st.session_state.y_test, tuned_pred):.4f}")
                    col2.metric("Precision", f"{precision_score(st.session_state.y_test, tuned_pred, zero_division=0):.4f}")
                    col3.metric("Recall", f"{recall_score(st.session_state.y_test, tuned_pred, zero_division=0):.4f}")
                    col4.metric("F1 Score", f"{f1_score(st.session_state.y_test, tuned_pred, zero_division=0):.4f}")
                    col5.metric("ROC-AUC", f"{roc_auc_score(st.session_state.y_test, tuned_prob):.4f}")
        else:
            st.info("Hyperparameter tuning available for Random Forest and XGBoost only.")
            st.session_state.best_model = st.session_state.models[best_model_name]["model"]


# =============================================================================
# RESULTS PAGE
# =============================================================================

elif page == "📈 Results":
    st.header("📈 Model Results & Evaluation")

    if not st.session_state.trained:
        st.warning("⚠️ Please train models first!")
        st.stop()

    comparison = st.session_state.comparison
    models = st.session_state.models

    tab1, tab2, tab3, tab4 = st.tabs(["Comparison", "Confusion Matrix", "ROC Curves", "Feature Importance"])

    # --- Comparison Tab ---
    with tab1:
        st.subheader("Performance Metrics Comparison")

        fig = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        colors = px.colors.qualitative.Set1

        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison['Model'],
                y=comparison[metric],
                marker_color=colors[i % len(colors)]
            ))

        fig.update_layout(barmode='group', title='Model Performance Comparison',
                         yaxis_title='Score', xaxis_title='Model')
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        fig2 = go.Figure()
        for _, row in comparison.iterrows():
            fig2.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics],
                theta=metrics,
                fill='toself',
                name=row['Model']
            ))
        fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                          showlegend=True, title="Radar Chart Comparison")
        st.plotly_chart(fig2, use_container_width=True)

    # --- Confusion Matrix Tab ---
    with tab2:
        st.subheader("Confusion Matrices")

        selected_model = st.selectbox("Select Model", list(models.keys()))

        cm = confusion_matrix(st.session_state.y_test, models[selected_model]["predictions"])

        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                       labels=dict(x="Predicted", y="Actual"),
                       title=f"Confusion Matrix - {selected_model}")
        st.plotly_chart(fig, use_container_width=True)

        # Classification report
        report = classification_report(st.session_state.y_test, 
                                       models[selected_model]["predictions"],
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)

    # --- ROC Curves Tab ---
    with tab3:
        st.subheader("ROC Curves")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                name='Random Classifier', line=dict(dash='dash')))

        for name, model_data in models.items():
            fpr, tpr, _ = roc_curve(st.session_state.y_test, model_data["probabilities"])
            auc = roc_auc_score(st.session_state.y_test, model_data["probabilities"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                    name=f"{name} (AUC={auc:.3f})"))

        fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate",
                         yaxis_title="True Positive Rate", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Precision-Recall curves
        st.subheader("Precision-Recall Curves")
        fig2 = go.Figure()
        for name, model_data in models.items():
            precision, recall, _ = precision_recall_curve(st.session_state.y_test, model_data["probabilities"])
            fig2.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=name))

        fig2.update_layout(title="Precision-Recall Curves", xaxis_title="Recall",
                          yaxis_title="Precision")
        st.plotly_chart(fig2, use_container_width=True)

    # --- Feature Importance Tab ---
    with tab4:
        st.subheader("Feature Importance")

        if st.session_state.best_model is not None:
            model = st.session_state.best_model

            # Tree-based feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                imp_df = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False).head(20)

                fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                            title=f"Feature Importance - {type(model).__name__}")
                st.plotly_chart(fig, use_container_width=True)

            # Permutation importance
            st.subheader("Permutation Importance")
            with st.spinner("Computing permutation importance..."):
                result = permutation_importance(
                    model, st.session_state.X_test, st.session_state.y_test,
                    n_repeats=5, random_state=random_state, n_jobs=-1
                )
                perm_df = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Importance': result.importances_mean
                }).sort_values('Importance', ascending=False).head(20)

                fig = px.bar(perm_df, x='Importance', y='Feature', orientation='h',
                            title="Permutation Feature Importance (Top 20)")
                st.plotly_chart(fig, use_container_width=True)

            # SHAP
            if SHAP_AVAILABLE and st.checkbox("Show SHAP Summary"):
                with st.spinner("Computing SHAP values..."):
                    try:
                        explainer = shap.Explainer(model, st.session_state.X_train)
                        shap_values = explainer(st.session_state.X_test[:100])

                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, st.session_state.X_test[:100],
                                         feature_names=st.session_state.feature_names,
                                         show=False)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"SHAP computation failed: {str(e)}")
        else:
            st.info("Train and tune a model to see feature importance.")


# =============================================================================
# PREDICTION PAGE
# =============================================================================

elif page == "🔮 Predict":
    st.header("🔮 Make Predictions")

    if st.session_state.best_model is None:
        st.warning("⚠️ Please train and select a best model first!")
        st.stop()

    pred_tab1, pred_tab2 = st.tabs(["Batch Prediction (CSV)", "Single Prediction"])

    # --- Batch Prediction ---
    with pred_tab1:
        st.subheader("Upload CSV for Batch Prediction")
        pred_file = st.file_uploader("Upload prediction data", type=["csv"], key="pred_upload")

        if pred_file is not None:
            try:
                pred_df = pd.read_csv(pred_file, encoding="latin1")
                st.write(f"Uploaded {pred_df.shape[0]} records")
                st.dataframe(pred_df.head(), use_container_width=True)

                if st.button("🔮 Predict Batch", type="primary"):
                    with st.spinner("Processing predictions..."):
                        # Apply same preprocessing
                        pred_clean, _ = clean_data(pred_df, missing_threshold)
                        pred_eng = engineer_features(pred_clean)
                        pred_enc = encode_features(pred_eng)

                        # Align columns with training data
                        train_cols = st.session_state.feature_names
                        for col in train_cols:
                            if col not in pred_enc.columns:
                                pred_enc[col] = 0
                        pred_enc = pred_enc[train_cols]

                        # Scale and predict
                        X_pred = st.session_state.scaler.transform(pred_enc)
                        predictions = st.session_state.best_model.predict(X_pred)
                        probabilities = st.session_state.best_model.predict_proba(X_pred)[:, 1]

                        result_df = pred_df.copy()
                        result_df['Late_Delivery_Prediction'] = predictions
                        result_df['Late_Delivery_Probability'] = probabilities
                        result_df['Risk_Level'] = pd.cut(probabilities, 
                                                         bins=[0, 0.3, 0.7, 1],
                                                         labels=['Low', 'Medium', 'High'])

                        st.subheader("Prediction Results")
                        st.dataframe(result_df[['Late_Delivery_Prediction', 
                                                'Late_Delivery_Probability',
                                                'Risk_Level']].head(20), use_container_width=True)

                        # Download
                        csv = result_df.to_csv(index=False)
                        st.download_button("📥 Download Results", csv, 
                                          "predictions.csv", "text/csv")

                        # Distribution
                        fig = px.pie(result_df, names='Risk_Level', 
                                    title="Risk Level Distribution")
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

    # --- Single Prediction ---
    with pred_tab2:
        st.subheader("Manual Input Prediction")
        st.markdown("Enter order details to predict late delivery risk.")

        # Create input fields based on common features
        col1, col2, col3 = st.columns(3)

        with col1:
            days_scheduled = st.number_input("Days for Shipment (Scheduled)", min_value=1, max_value=30, value=4)
            order_qty = st.number_input("Order Item Quantity", min_value=1, max_value=100, value=1)
            product_price = st.number_input("Product Price", min_value=0.0, value=100.0)

        with col2:
            sales = st.number_input("Sales Amount", min_value=0.0, value=300.0)
            benefit = st.number_input("Benefit per Order", value=50.0)
            shipping_mode = st.selectbox("Shipping Mode", ["Standard Class", "First Class", "Second Class", "Same Day"])

        with col3:
            market = st.selectbox("Market", ["Pacific Asia", "Europe", "USCA", "LATAM", "Africa"])
            category = st.selectbox("Category", ["Sporting Goods", "Electronics", "Apparel", "Home & Garden"])
            order_month = st.selectbox("Order Month", list(range(1, 13)), index=0)

        if st.button("🔮 Predict Single", type="primary"):
            with st.spinner("Predicting..."):
                # Build a single-row dataframe matching training structure
                # This is simplified - in production you'd need full feature alignment
                input_data = {
                    'Days for shipment (scheduled)': [days_scheduled],
                    'Order Item Quantity': [order_qty],
                    'Order Item Product Price': [product_price],
                    'Sales': [sales],
                    'Benefit per order': [benefit],
                    'Shipping Mode': [shipping_mode],
                    'Market': [market],
                    'Category Name': [category],
                    'order date (DateOrders)': [f"2018-{order_month:02d}-15"],
                }

                input_df = pd.DataFrame(input_data)

                # Apply preprocessing pipeline
                input_clean, _ = clean_data(input_df, missing_threshold)
                input_eng = engineer_features(input_clean)
                input_enc = encode_features(input_eng)

                # Align
                train_cols = st.session_state.feature_names
                for col in train_cols:
                    if col not in input_enc.columns:
                        input_enc[col] = 0
                input_enc = input_enc[train_cols]

                X_single = st.session_state.scaler.transform(input_enc)
                pred = st.session_state.best_model.predict(X_single)[0]
                prob = st.session_state.best_model.predict_proba(X_single)[0, 1]

                # Display result
                if pred == 1:
                    st.error(f"⚠️ **LATE DELIVERY RISK DETECTED**")
                    st.error(f"Probability: {prob:.1%}")
                else:
                    st.success(f"✅ **ON-TIME DELIVERY EXPECTED**")
                    st.success(f"Late Delivery Probability: {prob:.1%}")

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Late Delivery Risk %"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "red" if pred == 1 else "green"},
                           'steps': [
                               {'range': [0, 30], 'color': "lightgreen"},
                               {'range': [30, 70], 'color': "yellow"},
                               {'range': [70, 100], 'color': "salmon"}],
                           'threshold': {'line': {'color': "black", 'width': 4},
                                        'thickness': 0.75, 'value': 50}}
                ))
                st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# ABOUT PAGE
# =============================================================================

elif page == "ℹ️ About":
    st.header("ℹ️ About This Application")

    st.markdown("""
    ### Late Delivery Prediction App

    This Streamlit application provides an end-to-end machine learning pipeline for 
    predicting late deliveries in supply chain operations using the **DataCo Supply Chain Dataset**.

    #### Features
    - **📤 Data Upload**: Support for CSV files with automatic type detection
    - **📊 EDA**: Interactive visualizations including distributions, correlations, and missing value analysis
    - **🔧 Preprocessing**: Automated cleaning, feature engineering, encoding, and SMOTE balancing
    - **🤖 Model Training**: Compare 5 algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
    - **📈 Evaluation**: ROC curves, confusion matrices, precision-recall curves, and feature importance
    - **🔮 Prediction**: Both batch (CSV upload) and single-record prediction modes

    #### Pipeline Steps
    1. **Data Cleaning**: Handle missing values (drop columns > threshold%, impute rest)
    2. **Feature Engineering**: Create date-based, monetary, and frequency features
    3. **Encoding**: Label encoding for binary, one-hot for categorical
    4. **Scaling**: StandardScaler for numerical features
    5. **Balancing**: SMOTE for class imbalance
    6. **Training**: 5-fold cross-validation with multiple algorithms
    7. **Tuning**: GridSearchCV for Random Forest and XGBoost
    8. **Explainability**: Permutation importance and SHAP values

    #### Technologies
    - **Streamlit** - Web interface
    - **Pandas/NumPy** - Data manipulation
    - **Scikit-learn** - ML models and evaluation
    - **XGBoost** - Gradient boosting
    - **Plotly** - Interactive visualizations
    - **SHAP** - Model explainability

    #### Dataset
    The DataCo Supply Chain Dataset contains order fulfillment data including:
    - Shipping schedules vs actual delivery times
    - Customer and product information
    - Sales and profit metrics
    - Geographic and market data

    #### Author
    TINA UZOH
    """)

    st.markdown("---")
    st.markdown("**Version**: 1.0 | **Last Updated**: August 2026")
