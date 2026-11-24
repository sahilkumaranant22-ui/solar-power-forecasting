from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Solar PV Forecasting",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Solar PV Forecasting Dashboard")
st.caption("HQLSTM, HQCNN और classical baselines के 5, 10 और 15 मिनट के results")


# Metrics from your Jupyter test output
metrics = pd.DataFrame([
    {"Model": "HQLSTM", "Forecast (minutes ahead)": 5,  "MAE (kW)": 142.193, "RMSE (kW)": 242.092, "R²": 0.745},
    {"Model": "HQLSTM", "Forecast (minutes ahead)": 10, "MAE (kW)": 174.082, "RMSE (kW)": 290.290, "R²": 0.631},
    {"Model": "HQLSTM", "Forecast (minutes ahead)": 15, "MAE (kW)": 192.944, "RMSE (kW)": 309.370, "R²": 0.581},
    {"Model": "HQCNN", "Forecast (minutes ahead)": 5,  "MAE (kW)": 191.629, "RMSE (kW)": 289.717, "R²": 0.635},
    {"Model": "HQCNN", "Forecast (minutes ahead)": 10, "MAE (kW)": 222.315, "RMSE (kW)": 327.357, "R²": 0.531},
    {"Model": "HQCNN", "Forecast (minutes ahead)": 15, "MAE (kW)": 230.385, "RMSE (kW)": 338.455, "R²": 0.499},
    {"Model": "LSTM baseline", "Forecast (minutes ahead)": 5,  "MAE (kW)": 135.975, "RMSE (kW)": 236.984, "R²": 0.756},
    {"Model": "LSTM baseline", "Forecast (minutes ahead)": 10, "MAE (kW)": 174.793, "RMSE (kW)": 288.675, "R²": 0.635},
    {"Model": "LSTM baseline", "Forecast (minutes ahead)": 15, "MAE (kW)": 192.314, "RMSE (kW)": 302.782, "R²": 0.599},
    {"Model": "CNN baseline", "Forecast (minutes ahead)": 5,  "MAE (kW)": 142.000, "RMSE (kW)": 229.096, "R²": 0.772},
    {"Model": "CNN baseline", "Forecast (minutes ahead)": 10, "MAE (kW)": 182.669, "RMSE (kW)": 286.724, "R²": 0.640},
    {"Model": "CNN baseline", "Forecast (minutes ahead)": 15, "MAE (kW)": 194.668, "RMSE (kW)": 302.062, "R²": 0.601},
])


# Sidebar filters
st.sidebar.header("Filters")

horizon = st.sidebar.selectbox(
    "Forecast horizon",
    [5, 10, 15],
    format_func=lambda value: f"{value} minutes",
)

all_models = metrics["Model"].unique().tolist()

selected_models = st.sidebar.multiselect(
    "Models",
    all_models,
    default=all_models,
)

filtered = metrics[
    (metrics["Forecast (minutes ahead)"] == horizon)
    & (metrics["Model"].isin(selected_models))
]

if filtered.empty:
    st.warning("Sidebar में कम से कम एक model चुनें।")
    st.stop()


# Summary metrics
best_mae = filtered.loc[filtered["MAE (kW)"].idxmin()]
best_rmse = filtered.loc[filtered["RMSE (kW)"].idxmin()]
best_r2 = filtered.loc[filtered["R²"].idxmax()]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Lowest MAE",
    best_mae["Model"],
    f'{best_mae["MAE (kW)"]:.1f} kW',
)

col2.metric(
    "Lowest RMSE",
    best_rmse["Model"],
    f'{best_rmse["RMSE (kW)"]:.1f} kW',
)

col3.metric(
    "Highest R²",
    best_r2["Model"],
    f'{best_r2["R²"]:.3f}',
)


# Comparison charts
st.subheader(f"{horizon}-minute model comparison")

left, right = st.columns(2)

with left:
    mae_chart = px.bar(
        filtered.sort_values("MAE (kW)"),
        x="Model",
        y="MAE (kW)",
        color="Model",
        title="MAE (lower is better)",
    )
    mae_chart.update_layout(showlegend=False)
    st.plotly_chart(mae_chart, use_container_width=True)

with right:
    rmse_chart = px.bar(
        filtered.sort_values("RMSE (kW)"),
        x="Model",
        y="RMSE (kW)",
        color="Model",
        title="RMSE (lower is better)",
    )
    rmse_chart.update_layout(showlegend=False)
    st.plotly_chart(rmse_chart, use_container_width=True)


# Full metrics table and download
st.subheader("All model results")
st.dataframe(metrics, use_container_width=True, hide_index=True)

st.download_button(
    label="Download metrics CSV",
    data=metrics.to_csv(index=False).encode("utf-8"),
    file_name="solar_model_metrics.csv",
    mime="text/csv",
)


# Test data upload
st.subheader("Test dataset")
st.write("एक या कई test sample CSV files upload करें।")

uploaded_files = st.file_uploader(
    "Upload test CSV file(s)",
    type=["csv"],
    accept_multiple_files=True,
)

default_path = Path(__file__).parent / "solar_pv_test_samples.csv"
test_data = None

try:
    if uploaded_files:
        frames = [
            pd.read_csv(file, parse_dates=["Input end time"])
            for file in uploaded_files
        ]
        test_data = pd.concat(frames, ignore_index=True)

    elif default_path.exists():
        test_data = pd.read_csv(
            default_path,
            parse_dates=["Input end time"],
        )

except Exception as error:
    st.error(f"CSV पढ़ने में समस्या हुई: {error}")


if test_data is not None and not test_data.empty:
    if "Input end time" in test_data.columns:
        test_data = test_data.sort_values("Input end time")

    st.caption(f"Loaded test samples: {len(test_data):,}")

    target_columns = [
        f"Actual +{minutes} min power (kW)"
        for minutes in [5, 10, 15]
        if f"Actual +{minutes} min power (kW)" in test_data.columns
    ]

    if target_columns:
        selected_target = st.selectbox(
            "Actual power horizon",
            target_columns,
            format_func=lambda name: name.replace("Actual ", ""),
        )

        max_rows = min(len(test_data), 668)

        # Slider needs min_value < max_value.
        # For a single-row CSV, show that row without a slider.
        if max_rows > 1:
            sample_count = st.slider(
                "Number of samples to show",
                min_value=1,
                max_value=max_rows,
                value=min(300, max_rows),
            )
        else:
            sample_count = max_rows
            st.caption("इस CSV में 1 test sample है।")

        chart_data = test_data.head(sample_count).copy()

        if "Input end time" in chart_data.columns:
            chart_data = chart_data.set_index("Input end time")

        chart_data = chart_data[[selected_target]].rename(
            columns={selected_target: "Actual power (kW)"}
        )

        st.line_chart(chart_data)
        st.dataframe(test_data.head(20), use_container_width=True)

    else:
        st.warning("CSV में expected actual power columns नहीं मिले।")

else:
    st.info(
        "Test CSV upload करें। अगर default CSV इस्तेमाल करनी है, "
        "तो उसे dashboard.py के साथ project folder में रखें।"
    )


st.caption(
    "Dashboard आपके दिए हुए test metrics दिखाता है; "
    "trained model weights इसमें शामिल नहीं हैं।"
)