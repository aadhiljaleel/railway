
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Railway AI | Failure Risk Detection",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "data/indian_railway_failure_detection_maintenance_v2.csv"
MODEL_PATH = "model_pipeline.pkl"
TARGET = "risk_score"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1220;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }

    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263244;
        padding: 18px;
        border-radius: 14px;
    }

    .hero {
        padding: 28px 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827, #172033);
        border: 1px solid #263244;
        margin-bottom: 22px;
    }

    .hero h1 {
        margin: 0;
        font-size: 36px;
        color: #f8fafc;
    }

    .hero p {
        margin: 8px 0 0;
        color: #9ca3af;
        font-size: 16px;
    }

    .risk-card {
        padding: 25px;
        border-radius: 18px;
        background: #111827;
        border: 1px solid #263244;
        text-align: center;
        margin-top: 15px;
    }

    .risk-number {
        font-size: 58px;
        font-weight: 800;
        margin: 5px 0;
    }

    .risk-label {
        font-size: 18px;
        font-weight: 700;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 12px;
        color: #f8fafc;
    }

    .small-note {
        color: #9ca3af;
        font-size: 13px;
    }

    div.stButton > button {
        border-radius: 10px;
        height: 48px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA
# ============================================================

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        # Also allow the CSV to be beside app.py.
        alt = os.path.basename(DATA_PATH)
        if os.path.exists(alt):
            return pd.read_csv(alt)
        raise FileNotFoundError(
            f"Dataset not found. Put the CSV at '{DATA_PATH}'."
        )

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().copy()

    if TARGET not in df.columns:
        raise ValueError(
            f"'{TARGET}' was not found. Available columns: {list(df.columns)}"
        )

    # Target must be numeric for regression.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    return df


# ============================================================
# MODEL PIPELINE
# ============================================================

def make_pipeline(X):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


@st.cache_resource
def train_or_load_model(df):
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    pipeline = make_pipeline(X)

    pipeline.fit(X_train, y_train)

    # Save the COMPLETE pipeline, not just the model.
    # This prevents frontend encoding/scaling mismatch.
    joblib.dump(pipeline, MODEL_PATH)

    return pipeline


@st.cache_data
def model_metrics(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=42,
            max_depth=12,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(kernel="rbf"),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42,
            n_estimators=100,
        ),
    }

    rows = []

    for name, model in models.items():
        pipe = make_pipeline(X)
        pipe.set_params(model=model)
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        rows.append(
            {
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R²": r2,
            }
        )

    return pd.DataFrame(rows).sort_values("R²", ascending=False)


# ============================================================
# RISK GAUGE
# ============================================================

def risk_level(score):
    # These are presentation bands for the dashboard.
    # They do not change the model's prediction.
    if score < 40:
        return "LOW RISK", "low"
    if score < 70:
        return "MEDIUM RISK", "medium"
    return "HIGH RISK", "high"


def show_gauge(score):
    label, _ = risk_level(score)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score),
            number={"font": {"size": 42}},
            title={"text": label},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.25},
                "steps": [
                    {"range": [0, 40]},
                    {"range": [40, 70]},
                    {"range": [70, 100]},
                ],
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.8,
                    "value": float(score),
                },
            },
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# INPUT FORM
# ============================================================

def input_form(df):
    features = [c for c in df.columns if c != TARGET]

    # Keep the form manageable by showing all features in two columns.
    values = {}

    left, right = st.columns(2)

    for i, col in enumerate(features):
        container = left if i % 2 == 0 else right
        s = df[col]

        with container:
            if pd.api.types.is_numeric_dtype(s):
                clean = pd.to_numeric(s, errors="coerce").dropna()

                if clean.empty:
                    values[col] = 0.0
                    continue

                mn = float(clean.min())
                mx = float(clean.max())
                med = float(clean.median())

                if mn == mx:
                    values[col] = st.number_input(
                        col,
                        value=med,
                        disabled=True,
                    )
                else:
                    values[col] = st.number_input(
                        col,
                        min_value=mn,
                        max_value=mx,
                        value=med,
                    )
            else:
                options = (
                    s.dropna()
                    .astype(str)
                    .value_counts()
                    .index
                    .tolist()
                )

                if not options:
                    options = ["Unknown"]

                values[col] = st.selectbox(
                    col,
                    options,
                    index=0,
                )

    return values


# ============================================================
# PAGES
# ============================================================

def dashboard_page(df, model):
    st.markdown(
        """
        <div class="hero">
            <h1>🚆 Railway Failure Detection</h1>
            <p>AI-powered predictive maintenance and railway risk monitoring dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Railway Records", f"{len(df):,}")

    with c2:
        st.metric("Average Risk Score", f"{df[TARGET].mean():.2f}")

    with c3:
        st.metric("Maximum Risk", f"{df[TARGET].max():.2f}")

    with c4:
        high = (df[TARGET] >= 70).sum()
        st.metric("High-Risk Records", f"{high:,}")

    st.markdown('<div class="section-title">Risk Score Distribution</div>',
                unsafe_allow_html=True)

    fig = px.histogram(
        df,
        x=TARGET,
        nbins=30,
        marginal="box",
        title="Distribution of Predicted Target Variable",
    )
    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
    )
    st.plotly_chart(fig, use_container_width=True)

    numeric = df.select_dtypes(include=np.number)
    if TARGET in numeric.columns:
        corr = (
            numeric.corr(numeric_only=True)[TARGET]
            .drop(TARGET)
            .abs()
            .sort_values(ascending=False)
            .head(8)
            .sort_values()
        )

        st.markdown(
            '<div class="section-title">Features Most Related to Risk Score</div>',
            unsafe_allow_html=True,
        )

        fig2 = px.bar(
            x=corr.values,
            y=corr.index,
            orientation="h",
            labels={"x": "Absolute correlation", "y": "Feature"},
        )
        fig2.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
        )
        st.plotly_chart(fig2, use_container_width=True)


def prediction_page(df, model):
    st.markdown(
        """
        <div class="hero">
            <h1>🎯 Risk Prediction</h1>
            <p>Enter railway condition and operational details to estimate the risk score.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("risk_prediction_form"):
        values = input_form(df)
        submitted = st.form_submit_button(
            "🚨 PREDICT RISK SCORE",
            use_container_width=True,
        )

    if submitted:
        record = pd.DataFrame([values])

        try:
            score = float(model.predict(record)[0])

            # Keep the displayed score within the target's observed range.
            score = float(
                np.clip(score, df[TARGET].min(), df[TARGET].max())
            )

            label, css_class = risk_level(score)

            left, right = st.columns([1, 1])

            with left:
                st.markdown(
                    f"""
                    <div class="risk-card">
                        <div class="small-note">PREDICTED RISK SCORE</div>
                        <div class="risk-number">{score:.2f}</div>
                        <div class="risk-label">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right:
                show_gauge(score)

            st.success(
                f"Prediction completed using the Random Forest regression pipeline. "
                f"Risk score: {score:.2f}"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


def eda_page(df):
    st.markdown(
        """
        <div class="hero">
            <h1>📊 EDA & Analytics</h1>
            <p>Interactive visual analysis of the railway dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Risk score distribution
    fig = px.histogram(
        df,
        x=TARGET,
        nbins=30,
        title="Risk Score Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Season vs risk
    season_cols = [
        c for c in df.columns
        if c.lower() == "season"
    ]

    if season_cols:
        col = season_cols[0]
        temp = df.groupby(col, dropna=False)[TARGET].mean().reset_index()
        fig = px.line(
            temp,
            x=col,
            y=TARGET,
            markers=True,
            title="Season vs Average Risk Score",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Train type vs risk
    type_cols = [
        c for c in df.columns
        if c.lower() in {"train_type", "train type"}
    ]

    if type_cols:
        col = type_cols[0]
        temp = df.groupby(col, dropna=False)[TARGET].mean().reset_index()
        fig = px.bar(
            temp,
            x=col,
            y=TARGET,
            title="Train Type vs Average Risk Score",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Automatically find likely distance and train-age columns.
    for title, keywords, chart_type in [
        (
            "Distance Travelled vs Risk Score",
            ["distance", "travel"],
            "scatter",
        ),
        (
            "Train Age vs Risk Score",
            ["train", "age"],
            "line",
        ),
    ]:
        candidates = [
            c for c in df.columns
            if all(k in c.lower() for k in keywords)
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        if candidates:
            col = candidates[0]
            temp = (
                df.groupby(col, dropna=False)[TARGET]
                .mean()
                .reset_index()
                .sort_values(col)
            )

            if chart_type == "scatter":
                fig = px.scatter(
                    temp,
                    x=col,
                    y=TARGET,
                    title=title,
                )
            else:
                fig = px.line(
                    temp,
                    x=col,
                    y=TARGET,
                    markers=True,
                    title=title,
                )

            st.plotly_chart(fig, use_container_width=True)


def models_page(df):
    st.markdown(
        """
        <div class="hero">
            <h1>🤖 Model Performance</h1>
            <p>Comparison of the six regression algorithms used in the project.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Training/comparing the six models..."):
        results = model_metrics(df)

    display = results.copy()
    for col in ["MAE", "RMSE", "R²"]:
        display[col] = display[col].round(4)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    best = results.iloc[0]

    st.success(
        f"🏆 Best model by R² in this run: **{best['Model']}** "
        f"(R² = {best['R²']:.4f})"
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            results.sort_values("MAE"),
            x="Model",
            y="MAE",
            title="MAE — Lower is Better",
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            results.sort_values("R²", ascending=False),
            x="Model",
            y="R²",
            title="R² Score — Higher is Better",
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Note: this page recomputes the six models using the same raw-data "
        "preprocessing pipeline used by the application. If your notebook used "
        "different hyperparameters, the displayed metrics can differ from the notebook."
    )


def about_page(df):
    st.markdown(
        """
        <div class="hero">
            <h1>ℹ️ About the Project</h1>
            <p>Railway Failure Detection & Predictive Maintenance</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Project workflow

        **Dataset → EDA → Preprocessing → Encoding → Scaling → Train/Test Split
        → Six Regression Models → Evaluation → Random Forest → Prediction**

        ### Target variable

        `risk_score`

        ### Evaluation metrics

        - **MAE:** Mean Absolute Error — lower is better.
        - **RMSE:** Root Mean Squared Error — lower is better and penalizes large errors.
        - **R²:** Coefficient of Determination — higher is better.

        ### Frontend

        The dashboard is built with **Streamlit** and uses Plotly for interactive
        visualizations.

        ### Important

        The application saves the **entire preprocessing + Random Forest pipeline**
        as `model_pipeline.pkl`. This prevents the common problem where the
        frontend encodes/scales input differently from the training process.
        """
    )

    st.write("Dataset shape:", df.shape)
    st.write("Target:", TARGET)


# ============================================================
# MAIN
# ============================================================

def main():
    try:
        df = load_data()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Train/load the final Random Forest pipeline.
    with st.spinner("Loading Railway AI model..."):
        model = train_or_load_model(df)

    st.sidebar.markdown("## 🚆 RAILWAY AI")
    st.sidebar.caption("Failure Detection & Predictive Maintenance")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Risk Prediction",
            "EDA & Analytics",
            "Model Performance",
            "About",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.success("● SYSTEM ONLINE")
    st.sidebar.caption(f"Target: {TARGET}")
    st.sidebar.caption(f"Records: {len(df):,}")

    if page == "Dashboard":
        dashboard_page(df, model)
    elif page == "Risk Prediction":
        prediction_page(df, model)
    elif page == "EDA & Analytics":
        eda_page(df)
    elif page == "Model Performance":
        models_page(df)
    else:
        about_page(df)


if __name__ == "__main__":
    main()
