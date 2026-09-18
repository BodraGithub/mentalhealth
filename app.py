import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Mental Health Analytics in the Tech Industry",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE_SET2 = px.colors.qualitative.Set2
PALETTE_SET1 = px.colors.qualitative.Set1
PALETTE_PASTEL = px.colors.qualitative.Pastel
PALETTE_SET3 = px.colors.qualitative.Set3

WORK_INTERFERE_ORDER = ["Never", "Rarely", "Sometimes", "Often", "Don't know"]


# -----------------------------------------------------------------------
# DATA LOADING & CLEANING
# -----------------------------------------------------------------------
def clean_gender(gender):
    if pd.isna(gender):
        return "Unknown"
    gender = str(gender).lower().strip()

    male_terms = [
        "male", "m", "man", "cis male", "mal", "male-ish", "maile",
        "malr", "cis man", "make", "guy (-ish) ^_^", "male leaning androgynous",
    ]
    female_terms = [
        "female", "f", "woman", "cis female", "femake", "female ",
        "cis-female/femme", "female (cis)", "femail",
    ]
    trans_terms = ["trans-female", "trans woman", "female (trans)"]

    if gender in male_terms or ("male" in gender and "female" not in gender and "trans" not in gender):
        return "Male"
    elif gender in female_terms or ("female" in gender and "trans" not in gender and "male" not in gender):
        return "Female"
    elif gender in trans_terms or "trans" in gender:
        return "Trans"
    else:
        return "Other"


@st.cache_data
def load_data(path="survey.csv"):
    df = pd.read_csv(path)
    df_clean = df.copy()

    # Age cleaning: keep realistic ages only
    df_clean = df_clean[(df_clean["Age"] >= 15) & (df_clean["Age"] <= 75)]

    # Gender standardization
    df_clean["Gender"] = df_clean["Gender"].apply(clean_gender)

    # Missing value handling
    df_clean["self_employed"] = df_clean["self_employed"].fillna("No")
    df_clean["work_interfere"] = df_clean["work_interfere"].fillna("Don't know")
    if "comments" in df_clean.columns:
        df_clean = df_clean.drop(columns=["comments"])
    if "state" in df_clean.columns:
        df_clean["state"] = df_clean["state"].fillna("N/A")

    # Age groups
    bins = [15, 25, 35, 45, 55, 75]
    labels = ["15-24", "25-34", "35-44", "45-54", "55+"]
    df_clean["Age_Group"] = pd.cut(df_clean["Age"], bins=bins, labels=labels)

    return df_clean


df_clean = load_data()

# -----------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------
st.sidebar.title("🧠 Filters")
st.sidebar.markdown("Narrow the dataset down before exploring the charts.")

age_min, age_max = int(df_clean["Age"].min()), int(df_clean["Age"].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

genders = sorted(df_clean["Gender"].unique().tolist())
selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)

top_countries_all = df_clean["Country"].value_counts().index.tolist()
selected_countries = st.sidebar.multiselect(
    "Country (leave empty for all)", top_countries_all, default=[]
)

company_sizes = df_clean["no_employees"].dropna().unique().tolist()
selected_sizes = st.sidebar.multiselect("Company size", company_sizes, default=company_sizes)

treatment_filter = st.sidebar.radio("Sought treatment?", ["All", "Yes", "No"], index=0)

remote_only = st.sidebar.checkbox("Remote workers only", value=False)

# Apply filters
mask = (
    df_clean["Age"].between(age_range[0], age_range[1])
    & df_clean["Gender"].isin(selected_genders)
    & df_clean["no_employees"].isin(selected_sizes)
)
if selected_countries:
    mask &= df_clean["Country"].isin(selected_countries)
if treatment_filter != "All":
    mask &= df_clean["treatment"] == treatment_filter
if remote_only:
    mask &= df_clean["remote_work"] == "Yes"

data = df_clean[mask]

st.sidebar.markdown(f"**{len(data):,}** respondents match your filters (of {len(df_clean):,} total).")

if data.empty:
    st.warning("No respondents match the current filters — try widening them in the sidebar.")
    st.stop()

# -----------------------------------------------------------------------
# HEADER + KEY METRICS
# -----------------------------------------------------------------------
st.title("🧠 Mental Health in Tech Survey Dashboard")
st.caption("Exploratory analysis of the OSMI Mental Health in Tech Survey (2014).")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Respondents", f"{len(data):,}")
m2.metric("Avg. Age", f"{data['Age'].mean():.1f}")
m3.metric("Sought Treatment", f"{(data['treatment'] == 'Yes').mean() * 100:.1f}%")
m4.metric("Family History", f"{(data['family_history'] == 'Yes').mean() * 100:.1f}%")
m5.metric("Remote Workers", f"{(data['remote_work'] == 'Yes').mean() * 100:.1f}%")

st.divider()

# -----------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------
tab_overview, tab_uni, tab_bi, tab_multi, tab_insights = st.tabs(
    ["Overview", "Univariate", "Bivariate", "Multivariate", "Key Insights"]
)

# ---------------- OVERVIEW ----------------
with tab_overview:
    st.subheader("Dataset Preview")
    st.dataframe(data.head(50), use_container_width=True)
    with st.expander("Column summary"):
        st.dataframe(data.describe(include="all").transpose(), use_container_width=True)

# ---------------- UNIVARIATE ----------------
with tab_uni:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            data, x="Age", nbins=30,
            title="Age Distribution of Survey Respondents",
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        gender_counts = data["Gender"].value_counts()
        fig = px.pie(
            values=gender_counts.values, names=gender_counts.index,
            title="Gender Distribution", color_discrete_sequence=PALETTE_SET2,
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        country_counts = data["Country"].value_counts().head(15)
        fig = px.bar(
            x=country_counts.values, y=country_counts.index, orientation="h",
            title="Top 15 Countries by Respondents",
            color=country_counts.values, color_continuous_scale="Blues",
        )
        fig.update_layout(xaxis_title="Count", yaxis_title="Country", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.histogram(
            data, x="no_employees", title="Distribution of Company Size",
            color_discrete_sequence=["#EF553B"],
        )
        fig.update_layout(xaxis_title="Number of Employees", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)

    with c5:
        treatment_counts = data["treatment"].value_counts()
        fig = px.pie(
            values=treatment_counts.values, names=treatment_counts.index,
            title="Have You Sought Treatment for Mental Health?",
            color_discrete_sequence=["#00CC96", "#AB63FA"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        fig = px.histogram(
            data, x="family_history", title="Family History of Mental Illness",
            color="family_history", color_discrete_sequence=PALETTE_SET1,
        )
        fig.update_layout(xaxis_title="Family History", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- BIVARIATE ----------------
with tab_bi:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            data, x="Gender", color="treatment", barmode="group",
            title="Treatment Seeking by Gender", color_discrete_sequence=PALETTE_SET2,
        )
        fig.update_layout(xaxis_title="Gender", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            data, x="family_history", color="treatment", barmode="group",
            title="Treatment by Family History", color_discrete_sequence=PALETTE_PASTEL,
        )
        fig.update_layout(xaxis_title="Family History", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.histogram(
            data, x="work_interfere", color="treatment", barmode="group",
            title="Work Interference by Treatment Status",
            category_orders={"work_interfere": WORK_INTERFERE_ORDER},
            color_discrete_sequence=PALETTE_SET3,
        )
        fig.update_layout(xaxis_title="Work Interference", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.histogram(
            data, x="no_employees", color="benefits", barmode="group",
            title="Mental Health Benefits by Company Size",
            color_discrete_sequence=PALETTE_SET1,
        )
        fig.update_layout(xaxis_title="Company Size", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)

    with c5:
        fig = px.histogram(
            data, x="mental_health_consequence", color="Gender", barmode="group",
            title="Perceived Mental Health Consequences by Gender",
            color_discrete_sequence=PALETTE_SET2,
        )
        fig.update_layout(xaxis_title="Mental Health Consequence", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        fig = px.histogram(
            data, x="supervisor", color="treatment", barmode="group",
            title="Willingness to Discuss with Supervisor",
            color_discrete_sequence=PALETTE_PASTEL,
        )
        fig.update_layout(xaxis_title="Willing to Discuss", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- MULTIVARIATE ----------------
with tab_multi:
    top5 = data["Country"].value_counts().head(5).index
    df_top = data[data["Country"].isin(top5)]

    if not df_top.empty:
        treatment_by_country_gender = (
            df_top.groupby(["Country", "Gender"])["treatment"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Treatment_Rate")
        )
        fig = px.bar(
            treatment_by_country_gender, x="Country", y="Treatment_Rate", color="Gender",
            barmode="group", title="Treatment Rate by Country and Gender (Top 5 Countries)",
            color_discrete_sequence=PALETTE_SET2,
        )
        fig.update_layout(yaxis_title="Treatment Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        cross_tab = pd.crosstab(data["benefits"], data["care_options"])
        fig = px.imshow(
            cross_tab, text_auto=True, aspect="auto",
            title="Benefits vs Care Options Heatmap", color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.sunburst(
            data, path=["no_employees", "benefits", "treatment"],
            title="Company Size → Benefits → Treatment Hierarchy",
            color="treatment", color_discrete_sequence=PALETTE_SET2,
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.box(
            data, x="treatment", y="Age", color="Gender",
            title="Age Distribution by Treatment and Gender",
            color_discrete_sequence=PALETTE_SET2,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.violin(
            data, x="work_interfere", y="Age", color="treatment", box=True,
            title="Age by Work Interference and Treatment",
            category_orders={"work_interfere": WORK_INTERFERE_ORDER},
            color_discrete_sequence=PALETTE_SET2,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap of Key Variables")
    df_encoded = data.copy()
    le = LabelEncoder()
    categorical_cols = df_encoded.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    corr_cols = [
        "Age", "family_history", "treatment", "work_interfere", "benefits",
        "care_options", "wellness_program", "seek_help", "anonymity",
        "mental_health_consequence", "phys_health_consequence", "coworkers",
        "supervisor", "mental_health_interview", "mental_vs_physical", "obs_consequence",
    ]
    corr_cols = [c for c in corr_cols if c in df_encoded.columns]
    corr_matrix = df_encoded[corr_cols].corr()
    fig = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto",
        title="Correlation Heatmap of Key Variables",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mental vs Physical Health Perception by Company Size (%)")
    mental_vs_physical = (
        pd.crosstab(data["no_employees"], data["mental_vs_physical"], normalize="index") * 100
    )
    fig = px.bar(
        mental_vs_physical, barmode="stack",
        title="Mental vs Physical Health Perception by Company Size (%)",
        color_discrete_sequence=PALETTE_SET3,
    )
    fig.update_layout(xaxis_title="Company Size", yaxis_title="Percentage (%)")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- KEY INSIGHTS ----------------
with tab_insights:
    st.subheader("Key Insights")
    insights = [
        f"**Total Respondents:** {len(data):,}",
        f"**Average Age:** {data['Age'].mean():.1f} years",
        f"**Gender Distribution:** {data['Gender'].value_counts().to_dict()}",
        f"**Treatment Rate:** {(data['treatment'] == 'Yes').mean() * 100:.1f}%",
        f"**Family History Rate:** {(data['family_history'] == 'Yes').mean() * 100:.1f}%",
        f"**Top Country:** {data['Country'].value_counts().index[0] if len(data) else 'N/A'}",
        f"**Self-Employed:** {(data['self_employed'] == 'Yes').mean() * 100:.1f}%",
        f"**Remote Workers:** {(data['remote_work'] == 'Yes').mean() * 100:.1f}%",
        f"**Tech Company Workers:** {(data['tech_company'] == 'Yes').mean() * 100:.1f}%",
        f"**Observed Negative Consequences:** {(data['obs_consequence'] == 'Yes').mean() * 100:.1f}%",
    ]
    for line in insights:
        st.markdown(f"- {line}")

st.divider()
st.caption("Built with Streamlit · Data: OSMI Mental Health in Tech Survey")
