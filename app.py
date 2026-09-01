import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "indian_railway_failure_detection_maintenance_v2.csv"
TARGET_COLUMN = "maintenance_required"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, accuracy, report


def get_inputs(df):
    user_input = {}
    features = [col for col in df.columns if col != TARGET_COLUMN]

    for col in features:
        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            if col == "train_id":
                value = int(series.median()) if not series.empty else 0
                user_input[col] = st.number_input(f"{col}", min_value=0, value=value, step=1)
            else:
                min_value = float(series.min()) if not series.empty else 0.0
                max_value = float(series.max()) if not series.empty else 100.0
                default = float(series.median()) if not series.empty else 0.0
                if max_value == min_value:
                    step = 1.0
                else:
                    step = (max_value - min_value) / 100
                user_input[col] = st.number_input(
                    f"{col}",
                    min_value=min_value,
                    max_value=max_value,
                    value=default,
                    step=step,
                )
        else:
            options = sorted(series.dropna().unique().tolist())
            user_input[col] = st.selectbox(f"{col}", options)

    return user_input


def main():
    st.set_page_config(page_title="Railway Maintenance Predictor", page_icon="🚆", layout="wide")
    st.title("🚆 Railway Maintenance Predictor")
    st.caption("Predict whether a train requires maintenance based on historical railway data.")

    df = load_data()
    if TARGET_COLUMN not in df.columns:
        st.error(f"Target column '{TARGET_COLUMN}' was not found in the dataset.")
        return

    model, accuracy, _ = train_model(df)

    col1, col2 = st.columns([1.7, 1])

    with col1:
        st.subheader("Train details")
        with st.form("train_form"):
            user_input = get_inputs(df)
            submitted = st.form_submit_button("Predict Maintenance", use_container_width=True)

        if submitted:
            record = pd.DataFrame([user_input])
            prediction = int(model.predict(record)[0])
            probabilities = model.predict_proba(record)[0]
            confidence = float(probabilities.max() * 100)

            if prediction == 1:
                st.error("⚠️ Maintenance Required")
                st.write("This train is likely to need maintenance soon.")
            else:
                st.success("✅ No Maintenance Required")
                st.write("This train appears to be operating normally.")

            st.info(f"Prediction confidence: {confidence:.1f}%")

    with col2:
        st.subheader("Model summary")
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")
        st.write("This model was trained on the project dataset and predicts maintenance needs using a Random Forest classifier.")

        st.subheader("Dataset preview")
        st.dataframe(df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
