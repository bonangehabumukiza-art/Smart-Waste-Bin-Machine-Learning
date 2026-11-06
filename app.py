
import json
import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st


BASE = Path(__file__).parent
ART = BASE / "artifacts"
DATA = BASE / "data"

KIGALI = dt.timezone(dt.timedelta(hours=2), "CAT")


def kigali_now():
    """Return the current Kigali time."""
    return dt.datetime.now(KIGALI).replace(tzinfo=None)


st.set_page_config(
    page_title="Smart Waste-Bin",
    page_icon="🗑️",
    layout="wide"
)


@st.cache_data
def load_metrics():
    path = ART / "metrics.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_dataset():
    """Load the cleaned dataset if it exists."""
    candidates = [
        DATA / "cleaned_waste_data.csv",
        DATA / "cleaned_dataset.csv",
        BASE / "cleaned_waste_data.csv",
    ]

    for path in candidates:
        if path.exists():
            return pd.read_csv(path), path.name

    return None, None


def predict_hours(metrics, fill, now, last_collection):
    """Predict hours until the configured collection level."""

    hours_since = max(
        (now - last_collection).total_seconds() / 3600,
        metrics.get("min_hours", 1)
    )

    features = {
        "fill_pct": fill,
        "hours_since_collection": hours_since,
        "avg_fill_rate": fill / hours_since,
        "hour": now.hour,
        "is_weekend": int(now.weekday() >= 5),
    }

    coefficients = metrics.get("coefficients", {})

    # Do not invent a weight value if the trained model requires one.
    unsupported = [
        name for name in coefficients
        if name not in features
    ]

    if unsupported:
        raise ValueError(
            "The saved model requires features not provided by this app: "
            + ", ".join(unsupported)
            + ". Retrain the model using the available sensor features."
        )

    prediction = metrics["intercept"] + sum(
        coefficients[name] * features[name]
        for name in coefficients
    )

    return max(float(prediction), 0.0), hours_since


st.title("🗑️ Smart Waste-Bin System")

st.caption(
    "Monitor ultrasonic bin fill levels and estimate when a bin "
    "will reach its collection threshold."
)

metrics_path = ART / "metrics.json"

if not metrics_path.exists():
    st.error(
        "Model metrics not found. Add artifacts/metrics.json "
        "to your GitHub repository."
    )
    st.stop()

try:
    metrics = load_metrics()
except Exception as exc:
    st.error(f"Could not load model metrics: {exc}")
    st.stop()


FULL_LEVEL = float(metrics.get("full_level_pct", 80))
SOON_LEVEL = max(FULL_LEVEL - 15, 0)

if "history" not in st.session_state:
    st.session_state.history = []


tab1, tab2, tab3 = st.tabs(
    ["Predict", "Model and data", "Dataset explorer"]
)


# --------------------------------------------------
# TAB 1: PREDICTION
# --------------------------------------------------

with tab1:
    st.subheader("Bin fill prediction")

    fill = st.number_input(
        "Current ultrasonic fill level (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=1.0
    )

    col1, col2 = st.columns(2)

    col1.metric("Current fill level", f"{fill:.1f}%")
    col2.metric("Collection threshold", f"{FULL_LEVEL:.0f}%")

    st.progress(
        min(fill / 100, 1.0),
        text=f"{fill:.1f}% of bin capacity"
    )

    st.markdown("**Collection timing**")
    st.caption(
        f"Current Kigali time: "
        f"{kigali_now():%a %d %b %Y, %H:%M}"
    )

    if "default_last" not in st.session_state:
        st.session_state.default_last = (
            kigali_now().replace(second=0, microsecond=0)
            - dt.timedelta(hours=24)
        )

    if "default_now_time" not in st.session_state:
        st.session_state.default_now_time = (
            kigali_now().time().replace(second=0, microsecond=0)
        )

    d1, d2 = st.columns(2)

    last_date = d1.date_input(
        "Last collection date",
        value=st.session_state.default_last.date()
    )

    last_time = d2.time_input(
        "Last collection time",
        value=st.session_state.default_last.time()
    )

    use_now = st.checkbox(
        "Use current Kigali time",
        value=True
    )

    if use_now:
        now = kigali_now()
    else:
        n1, n2 = st.columns(2)

        selected_date = n1.date_input(
            "Current date",
            value=kigali_now().date()
        )

        selected_time = n2.time_input(
            "Current time",
            value=st.session_state.default_now_time
        )

        now = dt.datetime.combine(selected_date, selected_time)

    last_collection = dt.datetime.combine(last_date, last_time)

    if st.button("Predict collection time", type="primary"):

        if last_collection >= now:
            st.error("The last collection must be before the current time.")

        else:
            remaining = max(FULL_LEVEL - fill, 0.0)
            free_space = max(100 - fill, 0.0)

            c1, c2 = st.columns(2)

            c1.metric(
                "Fill remaining before collection",
                f"{remaining:.1f}%"
            )

            c2.metric(
                "Total empty space",
                f"{free_space:.1f}%"
            )

            if fill >= FULL_LEVEL:
                st.error(
                    f"The bin has reached the {FULL_LEVEL:.0f}% "
                    "collection threshold. Collect it now."
                )

            elif fill >= SOON_LEVEL:
                st.warning(
                    "The bin is approaching the collection threshold. "
                    "Plan collection soon."
                )

            else:
                try:
                    hours, _ = predict_hours(
                        metrics,
                        fill,
                        now,
                        last_collection
                    )

                    whole_hours = int(hours)
                    minutes = int(round((hours % 1) * 60))

                    if minutes == 60:
                        whole_hours += 1
                        minutes = 0

                    st.success(
                        f"Estimated time until collection threshold: "
                        f"{whole_hours} hours and {minutes} minutes."
                    )

                    st.caption(
                        "This is a machine-learning estimate, "
                        "not a guaranteed collection time."
                    )

                except ValueError as exc:
                    st.error(str(exc))

            st.session_state.history.append({
                "Time": now.strftime("%d %b %Y %H:%M"),
                "Fill (%)": round(fill, 1),
                "Remaining (%)": round(remaining, 1),
            })

    if st.session_state.history:
        st.divider()
        st.markdown("**Prediction history for this session**")

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download session history (CSV)",
            history_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()


# --------------------------------------------------
# TAB 2: MODEL AND DATA
# --------------------------------------------------

with tab2:
    st.subheader("Model information")

    info1, info2, info3 = st.columns(3)

    info1.metric(
        "Dataset rows",
        f"{metrics.get('rows', 'N/A'):,}"
        if isinstance(metrics.get("rows"), int)
        else str(metrics.get("rows", "N/A"))
    )

    info2.metric(
        "Collection threshold",
        f"{FULL_LEVEL:.0f}%"
    )

    info3.metric(
        "Model",
        "Multiple Linear Regression"
    )

    st.markdown("**Data source**")
    st.write(metrics.get("data_source", "Not specified"))

    st.markdown("**Training information**")
    st.write(metrics.get("weight", "Not specified"))

    st.markdown("**Test-set evaluation**")

    evaluation = {
        "Baseline (fill % only)": metrics.get("baseline", {}),
        "Multiple Linear Regression": metrics.get("mlr", {}),
        "Ridge Regression": metrics.get("ridge", {}),
    }

    st.dataframe(
        pd.DataFrame(evaluation).T,
        use_container_width=True
    )

    st.markdown("**Regression coefficients**")

    coefficients = metrics.get("coefficients", {})

    if coefficients:
        coefficient_df = pd.DataFrame(
            list(coefficients.items()),
            columns=["Feature", "Coefficient"]
        )

        st.dataframe(
            coefficient_df,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        f"Model intercept: {metrics.get('intercept', 'N/A')}"
    )

    st.markdown("**Model evaluation charts**")

    for image_name in ("eda.png", "evaluation.png"):
        image_path = ART / image_name

        if image_path.exists():
            st.image(str(image_path), caption=image_name)

    st.caption(
        "The metrics and coefficients shown here are loaded from "
        "artifacts/metrics.json."
    )


# --------------------------------------------------
# TAB 3: DATASET EXPLORER
# --------------------------------------------------

with tab3:
    st.subheader("Cleaned dataset explorer")

    dataset, dataset_name = load_dataset()

    if dataset is None:
        st.warning(
            "No cleaned dataset was found. Add your cleaned CSV file "
            "to the data/ folder in your GitHub repository."
        )

    else:
        st.write(f"**Dataset file:** {dataset_name}")

        m1, m2, m3 = st.columns(3)

        m1.metric("Rows", f"{len(dataset):,}")
        m2.metric("Columns", len(dataset.columns))
        m3.metric(
            "Missing values",
            f"{int(dataset.isna().sum().sum()):,}"
        )

        st.markdown("**Preview of the dataset**")

        st.dataframe(
            dataset.head(100),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download cleaned dataset (CSV)",
            dataset.to_csv(index=False),
            file_name="cleaned_waste_data.csv",
            mime="text/csv"
        )

        st.divider()
        st.markdown("**Explore columns**")

        selected_column = st.selectbox(
            "Choose a column",
            options=list(dataset.columns)
        )

        st.write(f"**Column:** {selected_column}")

        st.write(
            dataset[selected_column].describe(
                include="all"
            ).to_frame()
        )

        if pd.api.types.is_numeric_dtype(dataset[selected_column]):
            st.markdown("**Column distribution**")

            chart_data = dataset[selected_column].dropna()

            if not chart_data.empty:
                st.bar_chart(chart_data.value_counts().sort_index())

        st.markdown("**Filter dataset**")

        filter_column = st.selectbox(
            "Choose a column to filter",
            options=list(dataset.columns),
            key="filter_column"
        )

        filter_values = (
            dataset[filter_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(filter_values) <= 200:
            selected_values = st.multiselect(
                "Select values",
                options=sorted(filter_values)
            )

            if selected_values:
                filtered = dataset[
                    dataset[filter_column].astype(str).isin(selected_values)
                ]
            else:
                filtered = dataset

        else:
            st.info(
                "This column has many unique values. "
                "Showing the unfiltered dataset."
            )
            filtered = dataset

        st.write(f"Showing {len(filtered):,} rows.")

        st.dataframe(
            filtered.head(500),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download filtered data (CSV)",
            filtered.to_csv(index=False),
            file_name="filtered_waste_data.csv",
            mime="text/csv"
        )
