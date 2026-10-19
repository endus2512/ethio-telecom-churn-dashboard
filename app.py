"""
app.py
------
Ethio Telecom Customer Churn Analytics Dashboard

A production-ready Streamlit application featuring interactive filters,
KPI metrics, Plotly visualizations, CSV export, and a customer churn
risk predictor powered by a lightweight scikit-learn model.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from generate_data import CSV_PATH, DB_PATH, generate_customers, load_from_sqlite, save_to_csv, save_to_sqlite

# ---------------------------------------------------------------------------
# Brand / theme constants (Ethio Telecom-inspired palette)
# ---------------------------------------------------------------------------
ETHIO_GREEN = "#00813D"
ETHIO_ORANGE = "#FF6600"

PLOTLY_COLORWAY = [ETHIO_GREEN, ETHIO_ORANGE, "#36A2AE", "#8C5AA7", "#D9A400", "#4169E1"]

PAGE_CONFIG = {
    "page_title": "Ethio Telecom Churn Analytics",
    "page_icon": "📡",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading customer data...")
def load_data() -> pd.DataFrame:
    """
    Load customers from SQLite (preferred) or CSV.
    If neither exists, generate a fresh synthetic dataset on the fly.
    """
    try:
        if DB_PATH.exists():
            df = load_from_sqlite()
        elif CSV_PATH.exists():
            df = pd.read_csv(CSV_PATH)
        else:
            df = generate_customers()
            save_to_csv(df)
            save_to_sqlite(df)
    except Exception as exc:  # noqa: BLE001 — surface friendly UI error
        st.error(f"Failed to load data: {exc}")
        df = generate_customers()

    return clean_dataframe(df)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and handle missing values gracefully."""
    df = df.copy()

    # Ensure expected columns exist
    required = [
        "Customer_ID",
        "Region",
        "Contract_Type",
        "Monthly_Bill_ETB",
        "Internet_Service",
        "telebirr_User",
        "Customer_Service_Calls",
        "Tenure_Months",
        "Churn",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {missing_cols}")

    # Categorical fills
    df["Region"] = df["Region"].fillna("Unknown").astype(str)
    df["Contract_Type"] = df["Contract_Type"].fillna("Prepaid").astype(str)
    df["Internet_Service"] = df["Internet_Service"].fillna("Unknown").astype(str)
    df["telebirr_User"] = df["telebirr_User"].fillna("No").astype(str)
    df["Churn"] = df["Churn"].fillna("No").astype(str)

    # Numeric fills (median for bill; 0 / mode-like defaults for counts)
    df["Monthly_Bill_ETB"] = pd.to_numeric(df["Monthly_Bill_ETB"], errors="coerce")
    bill_median = df["Monthly_Bill_ETB"].median()
    if pd.isna(bill_median):
        bill_median = 350.0
    df["Monthly_Bill_ETB"] = df["Monthly_Bill_ETB"].fillna(bill_median)

    df["Customer_Service_Calls"] = (
        pd.to_numeric(df["Customer_Service_Calls"], errors="coerce").fillna(0).astype(int)
    )
    df["Tenure_Months"] = (
        pd.to_numeric(df["Tenure_Months"], errors="coerce").fillna(1).astype(int)
    )

    df["Customer_ID"] = df["Customer_ID"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Churn risk model (trained once per data snapshot)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training churn risk model...")
def train_churn_model(df: pd.DataFrame):
    """Fit a compact Gradient Boosting classifier for risk scoring."""
    feature_cols = [
        "Tenure_Months",
        "Monthly_Bill_ETB",
        "Customer_Service_Calls",
        "Contract_Type",
        "Internet_Service",
        "telebirr_User",
        "Region",
    ]
    X = df[feature_cols]
    y = (df["Churn"] == "Yes").astype(int)

    categorical = ["Contract_Type", "Internet_Service", "telebirr_User", "Region"]
    numeric = ["Tenure_Months", "Monthly_Bill_ETB", "Customer_Service_Calls"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=80,
                    max_depth=3,
                    learning_rate=0.08,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model, feature_cols


def rule_based_risk_score(
    tenure: int,
    monthly_bill: float,
    service_calls: int,
    contract: str,
    telebirr: str,
    internet: str,
) -> float:
    """
    Transparent fallback score (0–1) used alongside the ML model
    so the predictor remains explainable in interviews/demos.
    """
    score = 0.15
    if tenure < 6:
        score += 0.25
    elif tenure < 12:
        score += 0.12
    if service_calls >= 4:
        score += 0.28
    elif service_calls >= 2:
        score += 0.12
    if monthly_bill > 700:
        score += 0.15
    elif monthly_bill > 450:
        score += 0.08
    if contract == "Prepaid":
        score += 0.10
    if telebirr == "No":
        score += 0.08
    if internet == "3G":
        score += 0.10
    if internet == "Fiber":
        score -= 0.08
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Apply Ethio Telecom-inspired styling."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            max-width: 100%;
            overflow-x: clip;
        }}
        [data-testid="stSidebar"] {{
            border-right: 1px solid rgba(128, 128, 128, 0.25);
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            color: inherit;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: inherit;
            font-weight: 600;
        }}
        h1, h2, h3 {{
            color: inherit;
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }}
        [data-testid="stMetric"] {{
            box-sizing: border-box;
            min-width: 0;
            border: 1px solid rgba(128, 128, 128, 0.3);
            border-left: 5px solid {ETHIO_GREEN};
            border-radius: 0.75rem;
            padding: 12px 16px;
            box-shadow: 0 2px 8px rgba(10, 37, 64, 0.08);
        }}
        [data-testid="stMetric"] label,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {{
            color: inherit;
        }}
        .ethio-banner {{
            background: linear-gradient(110deg, #00652f 0%, {ETHIO_GREEN} 78%);
            border-bottom: 4px solid {ETHIO_ORANGE};
            color: #ffffff;
            padding: 1.4rem 1.6rem;
            border-radius: 0.9rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 16px rgba(10, 37, 64, 0.12);
        }}
        .ethio-banner h1 {{
            color: #ffffff;
            margin: 0;
            font-size: 1.75rem;
        }}
        .ethio-banner p {{
            margin: 0.35rem 0 0 0;
            color: #ffffff;
            font-size: 0.98rem;
        }}
        [data-testid="stTabs"] [role="tab"] {{
            color: inherit;
        }}
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
            color: inherit;
            border-bottom-color: {ETHIO_GREEN};
        }}
        [data-testid="stSidebar"] [data-tag] {{
            color: #ffffff;
            background-color: {ETHIO_GREEN};
        }}
        [data-testid="stSidebar"] [data-baseweb="select"],
        [data-testid="stSidebar"] [data-baseweb="input"] {{
            max-width: 100%;
            color: inherit;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"]:focus-within,
        [data-testid="stSidebar"] [data-baseweb="input"]:focus-within {{
            border-color: {ETHIO_GREEN};
        }}
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section {{
            box-sizing: border-box;
            max-width: 100%;
            color: inherit;
            border: 1px dashed rgba(128, 128, 128, 0.4);
            border-radius: 0.75rem;
        }}
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button,
        [data-testid="stSidebar"] button[kind="secondary"] {{
            max-width: 100%;
            color: inherit;
            border-color: rgba(128, 128, 128, 0.4);
            white-space: normal;
        }}
        [data-testid="stSidebar"] button[kind="primary"],
        [data-testid="stFormSubmitButton"] button {{
            color: #ffffff;
            background-color: {ETHIO_GREEN};
            border-color: {ETHIO_GREEN};
        }}
        [data-testid="stDownloadButton"] button {{
            color: #1f2328;
            background-color: {ETHIO_ORANGE};
            border-color: {ETHIO_ORANGE};
        }}
        [data-testid="stDataFrame"] {{
            box-sizing: border-box;
            max-width: 100%;
            color: inherit;
            border: 1px solid rgba(128, 128, 128, 0.3);
            border-radius: 0.75rem;
            overflow: hidden;
        }}
        @media (max-width: 640px) {{
            [data-testid="stMainBlockContainer"] {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}
            .ethio-banner {{
                padding: 1rem;
                margin-bottom: 0.75rem;
                overflow-wrap: anywhere;
            }}
            .ethio-banner h1 {{
                font-size: clamp(1.25rem, 6vw, 1.6rem);
                line-height: 1.25;
            }}
            .ethio-banner p {{
                font-size: 0.9rem;
                line-height: 1.45;
            }}
            [data-testid="stMetric"] {{
                width: 100%;
                min-width: 0;
                padding: 0.75rem 1rem;
            }}
            [data-testid="stMetricLabel"],
            [data-testid="stMetricValue"] {{
                min-width: 0;
                overflow-wrap: anywhere;
            }}
            [data-testid="stSidebar"],
            [data-testid="stSidebar"] > div,
            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
                min-width: 0;
                max-width: 100%;
            }}
            [data-testid="stSidebar"] [data-testid="stFileUploader"],
            [data-testid="stSidebar"] [data-testid="stFileUploader"] section,
            [data-testid="stSidebar"] [data-testid="stFileUploader"] small {{
                box-sizing: border-box;
                max-width: 100%;
                overflow-wrap: anywhere;
            }}
            [data-testid="stSidebar"] [data-tag] {{
                max-width: 100%;
                white-space: normal;
                overflow-wrap: anywhere;
            }}
            [data-testid="stSidebar"] button,
            [data-testid="stDownloadButton"] button,
            [data-testid="stFormSubmitButton"] button {{
                max-width: 100%;
                white-space: normal;
                overflow-wrap: anywhere;
            }}
            [data-testid="stTabs"] [role="tab"] {{
                min-width: 0;
                padding-left: 0.45rem;
                padding-right: 0.45rem;
                white-space: normal;
                text-align: center;
            }}
            [data-testid="stDataFrame"] {{
                width: 100%;
                max-width: 100%;
            }}
        }}
        .risk-high {{
            border: 1px solid rgba(128, 128, 128, 0.3);
            border-left: 5px solid {ETHIO_ORANGE};
            padding: 1rem 1.2rem;
            border-radius: 0.75rem;
            color: inherit;
        }}
        .risk-medium {{
            border: 1px solid rgba(128, 128, 128, 0.3);
            border-left: 5px solid {ETHIO_ORANGE};
            padding: 1rem 1.2rem;
            border-radius: 0.75rem;
            color: inherit;
        }}
        .risk-low {{
            border: 1px solid rgba(128, 128, 128, 0.3);
            border-left: 5px solid {ETHIO_GREEN};
            padding: 1rem 1.2rem;
            border-radius: 0.75rem;
            color: inherit;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="ethio-banner">
            <h1>Ethio Telecom · Customer Churn Analytics</h1>
            <p>Monitor retention risk across regions, contracts, and digital services — portfolio demo with synthetic data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Sidebar filters for Region, Contract, Internet Service, and Churn."""
    st.sidebar.markdown("### 🔍 Filters")
    st.sidebar.caption("Narrow the dashboard to the segments you care about.")

    regions = sorted(df["Region"].unique().tolist())
    contracts = sorted(df["Contract_Type"].unique().tolist())
    internets = sorted(df["Internet_Service"].unique().tolist())
    churn_opts = sorted(df["Churn"].unique().tolist())

    selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
    selected_contracts = st.sidebar.multiselect("Contract Type", contracts, default=contracts)
    selected_internet = st.sidebar.multiselect("Internet Service", internets, default=internets)
    selected_churn = st.sidebar.multiselect("Churn Status", churn_opts, default=churn_opts)

    # Empty selection → empty frame (avoid silent full-data fallback)
    if not (selected_regions and selected_contracts and selected_internet and selected_churn):
        st.sidebar.warning("Select at least one value in each filter.")
        return df.iloc[0:0].copy()

    mask = (
        df["Region"].isin(selected_regions)
        & df["Contract_Type"].isin(selected_contracts)
        & df["Internet_Service"].isin(selected_internet)
        & df["Churn"].isin(selected_churn)
    )
    return df.loc[mask].copy()


def render_kpis(df: pd.DataFrame) -> None:
    """Top-level metrics."""
    total = len(df)
    churned = int((df["Churn"] == "Yes").sum()) if total else 0
    churn_rate = (churned / total * 100) if total else 0.0
    avg_bill = float(df["Monthly_Bill_ETB"].mean()) if total else 0.0

    c1, c2, c3, c4 = st.columns(4, gap="small", wrap=True)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Churn Rate", f"{churn_rate:.1f}%")
    c4.metric("Avg Monthly Bill", f"{avg_bill:,.0f} ETB")


def chart_churn_by_region(df: pd.DataFrame) -> go.Figure:
    """Interactive bar chart: churn rate (%) by region."""
    if df.empty:
        return go.Figure()

    summary = (
        df.groupby("Region", as_index=False)
        .agg(
            Customers=("Customer_ID", "count"),
            Churned=("Churn", lambda s: (s == "Yes").sum()),
        )
    )
    summary["Churn_Rate_%"] = (summary["Churned"] / summary["Customers"] * 100).round(1)
    summary = summary.sort_values("Churn_Rate_%", ascending=False)

    fig = px.bar(
        summary,
        x="Region",
        y="Churn_Rate_%",
        text="Churn_Rate_%",
        color="Churn_Rate_%",
        color_continuous_scale=[[0, ETHIO_GREEN], [1, ETHIO_ORANGE]],
        title="Churn Rate by Region (%)",
        hover_data={"Customers": True, "Churned": True, "Churn_Rate_%": True},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(t=60, b=40),
        yaxis_title="Churn Rate (%)",
        xaxis_title="",
    )
    return fig


def chart_internet_among_churned(df: pd.DataFrame) -> go.Figure:
    """Pie chart: internet service mix among churned customers."""
    churned = df[df["Churn"] == "Yes"]
    if churned.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Internet Service among Churned Customers",
            annotations=[
                dict(
                    text="No churned customers in current filter",
                    showarrow=False,
                    font=dict(size=14),
                )
            ],
        )
        return fig

    counts = churned["Internet_Service"].value_counts().reset_index()
    counts.columns = ["Internet_Service", "Count"]

    fig = px.pie(
        counts,
        names="Internet_Service",
        values="Count",
        title="Internet Service among Churned Customers",
        color_discrete_sequence=PLOTLY_COLORWAY,
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=20),
        legend_title_text="Service",
    )
    return fig


def chart_bill_vs_churn(df: pd.DataFrame) -> go.Figure:
    """Box plot of monthly bill by churn status (with underlying points)."""
    if df.empty:
        return go.Figure()

    fig = px.box(
        df,
        x="Churn",
        y="Monthly_Bill_ETB",
        color="Churn",
        points="outliers",
        title="Monthly Bill (ETB) vs Churn Status",
        color_discrete_map={"Yes": ETHIO_ORANGE, "No": ETHIO_GREEN},
        labels={"Monthly_Bill_ETB": "Monthly Bill (ETB)", "Churn": "Churned?"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=60, b=40),
    )
    return fig


def render_data_table(df: pd.DataFrame) -> None:
    """Filtered records + CSV download."""
    st.subheader("Customer Records")
    st.caption(f"Showing {len(df):,} filtered rows.")

    display_cols = [
        "Customer_ID",
        "Region",
        "Contract_Type",
        "Monthly_Bill_ETB",
        "Internet_Service",
        "telebirr_User",
        "Customer_Service_Calls",
        "Tenure_Months",
        "Churn",
    ]
    st.dataframe(
        df[display_cols].sort_values("Customer_ID"),
        width="stretch",
        height=360,
        alt="Filtered Ethio Telecom customer records",
    )

    csv_buffer = io.StringIO()
    df[display_cols].to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download filtered data (CSV)",
        data=csv_buffer.getvalue(),
        file_name="ethio_telecom_filtered_customers.csv",
        mime="text/csv",
        width="content",
    )


def render_risk_predictor(df: pd.DataFrame) -> None:
    """Individual customer churn risk scorer (ML + explainable rules)."""
    st.subheader("Individual Customer Risk Predictor")
    st.write(
        "Enter a customer's usage profile to estimate churn risk. "
        "The score blends a trained Gradient Boosting model with a transparent rule-based check."
    )

    model, feature_cols = train_churn_model(df)

    with st.form("risk_form"):
        col_a, col_b, col_c = st.columns(3, gap="medium", wrap=True)

        with col_a:
            tenure = st.number_input("Tenure (months)", min_value=1, max_value=120, value=8, step=1)
            monthly_bill = st.number_input(
                "Monthly Bill (ETB)", min_value=50.0, max_value=3000.0, value=480.0, step=10.0
            )
            service_calls = st.number_input(
                "Customer Service Calls", min_value=0, max_value=20, value=3, step=1
            )

        with col_b:
            contract = st.selectbox("Contract Type", sorted(df["Contract_Type"].unique()))
            internet = st.selectbox("Internet Service", sorted(df["Internet_Service"].unique()))
            telebirr = st.selectbox("telebirr User", ["Yes", "No"])

        with col_c:
            region = st.selectbox("Region", sorted(df["Region"].unique()))
            st.markdown("")  # spacing
            submitted = st.form_submit_button("Predict Churn Risk", width="stretch")

    if not submitted:
        st.info("Fill in the fields above and click **Predict Churn Risk**.")
        return

    row = pd.DataFrame(
        [
            {
                "Tenure_Months": int(tenure),
                "Monthly_Bill_ETB": float(monthly_bill),
                "Customer_Service_Calls": int(service_calls),
                "Contract_Type": contract,
                "Internet_Service": internet,
                "telebirr_User": telebirr,
                "Region": region,
            }
        ]
    )[feature_cols]

    ml_prob = float(model.predict_proba(row)[0][1])
    rule_prob = rule_based_risk_score(
        tenure=int(tenure),
        monthly_bill=float(monthly_bill),
        service_calls=int(service_calls),
        contract=contract,
        telebirr=telebirr,
        internet=internet,
    )
    # Blend: mostly ML, rules as a light prior for interpretability
    blended = 0.7 * ml_prob + 0.3 * rule_prob

    if blended >= 0.55:
        level, css = "HIGH RISK", "risk-high"
        advice = (
            "Prioritize retention outreach: offer a loyalty discount, upgrade network quality, "
            "or assign a care agent for a proactive check-in."
        )
    elif blended >= 0.35:
        level, css = "MEDIUM RISK", "risk-medium"
        advice = (
            "Monitor engagement. Promote telebirr / value bundles and reduce friction "
            "around billing or network complaints."
        )
    else:
        level, css = "LOW RISK", "risk-low"
        advice = "Customer appears stable. Focus on upsell (5G/Fiber) and long-term loyalty rewards."

    m1, m2, m3 = st.columns(3, gap="small", wrap=True)
    m1.metric("Blended Risk Score", f"{blended * 100:.1f}%")
    m2.metric("ML Model Probability", f"{ml_prob * 100:.1f}%")
    m3.metric("Rule-Based Score", f"{rule_prob * 100:.1f}%")

    st.markdown(
        f'<div class="{css}"><strong>{level}</strong><br/>{advice}</div>',
        unsafe_allow_html=True,
    )

    # Gauge for visual impact
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=blended * 100,
            number={"suffix": "%"},
            title={"text": "Churn Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": ETHIO_GREEN},
                "steps": [
                    {"range": [0, 35], "color": "#B7E3C7"},
                    {"range": [35, 55], "color": "#FFD6B8"},
                    {"range": [55, 100], "color": "#FFB27A"},
                ],
                "threshold": {
                    "line": {"color": ETHIO_ORANGE, "width": 3},
                    "thickness": 0.75,
                    "value": 55,
                },
            },
        )
    )
    gauge.update_layout(height=280, margin=dict(t=40, b=10, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(
        gauge,
        width="stretch",
        theme="streamlit",
        alt="Customer churn risk score",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(**PAGE_CONFIG)
    inject_custom_css()
    render_header()

    df = load_data()
    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload Custom Data")
    uploaded_file = st.sidebar.file_uploader(" Upload Ethio Telecom CSV ", type=["csv"])

    if uploaded_file is not None:
        try:
            df = clean_dataframe(pd.read_csv(uploaded_file))
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            st.sidebar.error(f"Could not load the uploaded CSV: {exc}")
            st.stop()

    filtered = apply_filters(df)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.caption(
        "Portfolio project using synthetic Ethio Telecom data. "
        f"Source: SQLite (`{DB_PATH.name}`) / CSV under `/data`."
    )
    if st.sidebar.button("🔄 Regenerate synthetic data"):
        with st.spinner("Generating new dataset..."):
            new_df = generate_customers()
            save_to_csv(new_df)
            save_to_sqlite(new_df)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    tab_overview, tab_predict, tab_data = st.tabs(
        ["📊 Overview & Insights", "🎯 Risk Predictor", "📋 Data & Export"]
    )

    with tab_overview:
        if filtered.empty:
            st.warning("No customers match the current filters. Adjust the sidebar selections.")
        else:
            render_kpis(filtered)
            st.markdown("")
            left, right = st.columns(2, gap="medium", wrap=True)
            with left:
                st.plotly_chart(
                    chart_churn_by_region(filtered),
                    width="stretch",
                    theme="streamlit",
                    alt="Churn rate by region",
                )
            with right:
                st.plotly_chart(
                    chart_internet_among_churned(filtered),
                    width="stretch",
                    theme="streamlit",
                    alt="Internet service mix among churned customers",
                )
            st.plotly_chart(
                chart_bill_vs_churn(filtered),
                width="stretch",
                theme="streamlit",
                alt="Monthly bill distribution by churn status",
            )

            # Quick SQL-flavored insight strip
            st.markdown("#### Snapshot Insights")
            i1, i2, i3 = st.columns(3, gap="small", wrap=True)
            top_region = (
                filtered.groupby("Region")["Churn"]
                .apply(lambda s: (s == "Yes").mean())
                .sort_values(ascending=False)
            )
            prepaid_rate = (
                (filtered.loc[filtered["Contract_Type"] == "Prepaid", "Churn"] == "Yes").mean()
                * 100
                if (filtered["Contract_Type"] == "Prepaid").any()
                else 0.0
            )
            avg_calls_churned = (
                filtered.loc[filtered["Churn"] == "Yes", "Customer_Service_Calls"].mean()
                if (filtered["Churn"] == "Yes").any()
                else 0.0
            )
            i1.info(
                f"**Highest churn region:** {top_region.index[0]} "
                f"({top_region.iloc[0] * 100:.1f}%)"
                if len(top_region)
                else "No region data"
            )
            i2.info(f"**Prepaid churn rate:** {prepaid_rate:.1f}%")
            i3.info(f"**Avg service calls (churned):** {avg_calls_churned:.1f}")

    with tab_predict:
        # Train on full (cleaned) population so filters don't bias the model
        render_risk_predictor(df)

    with tab_data:
        if filtered.empty:
            st.warning("No rows to display for the current filters.")
        else:
            render_data_table(filtered)

    st.markdown("---")
    st.caption(
        "Built for portfolio demonstration · Synthetic data only · "
        "Ethio Telecom branding colors used for thematic design (unofficial)."
    )


if __name__ == "__main__":
    main()
