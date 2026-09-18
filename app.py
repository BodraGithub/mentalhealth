import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import LabelEncoder

# =======================================================================
# PAGE CONFIG
# =======================================================================
st.set_page_config(
    page_title="Mental Health in Tech Survey",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =======================================================================
# COLOR PALETTES
# =======================================================================
LIGHT = {
    "bg": "#F8FAFC",
    "card": "#FFFFFF",
    "ink": "#0F172A",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "primary": "#0EA5E9",
    "primary_dark": "#0284C7",
    "accent": "#F43F5E",
    "success": "#10B981",
    "warning": "#F59E0B",
    "purple": "#8B5CF6",
    "pink": "#EC4899",
    "gradient": "linear-gradient(135deg, #667EEA 0%, #764BA2 100%)",
}

DARK = {
    "bg": "#0F172A",
    "card": "#1E293B",
    "ink": "#F1F5F9",
    "muted": "#CBD5E1",
    "border": "#334155",
    "primary": "#38BDF8",
    "primary_dark": "#0EA5E9",
    "accent": "#FB7185",
    "success": "#34D399",
    "warning": "#FBBF24",
    "purple": "#A78BFA",
    "pink": "#F472B6",
    "gradient": "linear-gradient(135deg, #667EEA 0%, #764BA2 100%)",
}

# Theme state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

C = DARK if st.session_state.dark_mode else LIGHT

# =======================================================================
# PLOTLY THEME
# =======================================================================
pio.templates["modern"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, -apple-system, sans-serif", size=13, color=C["ink"]),
        title_font=dict(family="Sora, Inter, sans-serif", size=16, color=C["ink"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[C["primary"], C["accent"], C["purple"], C["warning"], C["success"], C["pink"]],
        xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"], linecolor=C["border"]),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"], linecolor=C["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["ink"])),
        margin=dict(t=60, l=10, r=10, b=10),
        hoverlabel=dict(bgcolor=C["card"], font_size=12, font_family="Inter"),
    )
)
pio.templates.default = "modern"

SENTIMENT_MAP = {"Yes": C["primary"], "No": C["accent"], "Don't know": C["warning"], "Not sure": C["purple"]}
GENDER_MAP = {"Male": C["primary"], "Female": C["accent"], "Trans": C["warning"], "Other": C["success"], "Unknown": C["muted"]}
WORK_INTERFERE_ORDER = ["Never", "Rarely", "Sometimes", "Often", "Don't know"]
WORK_INTERFERE_MAP = {"Never": C["success"], "Rarely": C["primary"], "Sometimes": C["warning"], "Often": C["accent"], "Don't know": C["muted"]}

# =======================================================================
# CSS — High contrast, modern, readable
# =======================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Sora:wght@500;600;700;800&display=swap');

* {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

/* Main background */
.stApp {{
    background: {C["bg"]};
    color: {C["ink"]};
}}

/* Hide streamlit branding */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{background: transparent !important;}}

/* Typography */
h1, h2, h3, h4 {{
    font-family: 'Sora', sans-serif;
    color: {C["ink"]} !important;
    letter-spacing: -0.02em;
}}
h1 {{ font-weight: 800; }}
h2 {{ font-weight: 700; }}
h3 {{ font-weight: 600; }}

/* ============= SIDEBAR - HIGH CONTRAST FIX ============= */
[data-testid="stSidebar"] {{
    background: {C["card"]} !important;
    border-right: 1px solid {C["border"]};
}}

/* Force ALL sidebar text to be readable */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6 {{
    color: {C["ink"]} !important;
    font-weight: 700 !important;
}}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {{
    color: {C["ink"]} !important;
}}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown * {{
    color: {C["ink"]} !important;
}}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption p {{
    color: {C["muted"]} !important;
}}

/* Sidebar title with gradient */
[data-testid="stSidebar"] h1:first-of-type {{
    font-size: 1.15rem;
    font-weight: 700;
}}

[data-testid="stSidebar"] > div:first-child {{
    padding-top: 1.5rem;
}}

/* Multiselect tags */
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{
    background-color: {C["primary"]} !important;
    border-radius: 6px;
}}
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] span,
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] svg {{
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}}

/* Multiselect input */
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div {{
    background-color: {C["card"]} !important;
    border-color: {C["border"]} !important;
}}
[data-testid="stSidebar"] .stMultiSelect input {{
    color: {C["ink"]} !important;
}}

/* Dropdown menu */
[data-testid="stSidebar"] [role="listbox"],
[data-testid="stSidebar"] [data-baseweb="popover"],
[data-testid="stSidebar"] ul {{
    background-color: {C["card"]} !important;
}}
[data-testid="stSidebar"] [role="option"],
[data-testid="stSidebar"] li {{
    color: {C["ink"]} !important;
    background-color: {C["card"]} !important;
}}
[data-testid="stSidebar"] [role="option"]:hover,
[data-testid="stSidebar"] li:hover {{
    background-color: {C["primary"]}20 !important;
}}

/* Slider */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {{
    background-color: {C["primary"]} !important;
}}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {{
    color: {C["ink"]} !important;
}}
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {{
    color: {C["ink"]} !important;
}}

/* Radio buttons */
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio div,
[data-testid="stSidebar"] .stRadio p {{
    color: {C["ink"]} !important;
}}

/* Checkbox */
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stCheckbox div,
[data-testid="stSidebar"] .stCheckbox p {{
    color: {C["ink"]} !important;
}}

/* Toggle */
[data-testid="stSidebar"] .stToggle label,
[data-testid="stSidebar"] .stToggle p {{
    color: {C["ink"]} !important;
}}

/* Sidebar divider */
[data-testid="stSidebar"] hr {{
    border-color: {C["border"]};
    margin: 1rem 0;
}}

/* ============= HERO BANNER ============= */
.hero {{
    position: relative;
    background: linear-gradient(135deg, #667EEA 0%, #764BA2 50%, #F093FB 100%);
    background-size: 200% 200%;
    animation: gradientShift 8s ease infinite;
    border-radius: 20px;
    padding: 2.5rem 2.75rem;
    margin-bottom: 1.75rem;
    color: #FFFFFF;
    overflow: hidden;
    box-shadow: 0 20px 40px -15px rgba(102, 126, 234, 0.5);
}}
@keyframes gradientShift {{
    0%, 100% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
}}
.hero::before {{
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
}}
@keyframes float {{
    0%, 100% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-20px); }}
}}
.hero-eyebrow {{
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.9);
    margin-bottom: 0.75rem;
}}
.hero-title {{
    font-family: 'Sora', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0 0 0.5rem 0;
    color: #FFFFFF !important;
    letter-spacing: -0.03em;
    line-height: 1.1;
}}
.hero-sub {{
    font-size: 1.05rem;
    color: rgba(255,255,255,0.95) !important;
    max-width: 60ch;
    line-height: 1.6;
}}
.hero-number {{
    font-family: 'Sora', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    color: #FFFFFF !important;
    line-height: 1;
    letter-spacing: -0.03em;
}}
.hero-number-label {{
    color: rgba(255,255,255,0.9) !important;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

/* ============= STAT CARDS ============= */
.stat-card {{
    background: {C["card"]};
    border: 1px solid {C["border"]};
    border-radius: 16px;
    padding: 1.25rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.stat-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 24px -8px rgba(0,0,0,0.15);
    border-color: var(--accent, {C["primary"]});
}}
.stat-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: var(--accent, {C["primary"]});
    border-radius: 16px 0 0 16px;
}}
.stat-icon {{
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    opacity: 0.9;
}}
.stat-value {{
    font-family: 'Sora', sans-serif;
    font-size: 1.85rem;
    font-weight: 800;
    color: {C["ink"]} !important;
    line-height: 1;
    letter-spacing: -0.03em;
}}
.stat-label {{
    font-size: 0.8rem;
    color: {C["muted"]} !important;
    margin-top: 0.4rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}}

/* ============= SIDEBAR STAT ============= */
.sidebar-stat {{
    background: linear-gradient(135deg, {C["primary"]}20, {C["purple"]}20);
    border: 1px solid {C["primary"]}40;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin: 1rem 0;
}}
.sidebar-stat-value {{
    font-family: 'Sora', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: {C["primary"]} !important;
}}
.sidebar-stat-label {{
    font-size: 0.75rem;
    color: {C["muted"]} !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}}

/* ============= TABS ============= */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: {C["card"]};
    padding: 6px;
    border-radius: 12px;
    border: 1px solid {C["border"]};
    margin-bottom: 1.5rem;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    border-radius: 8px;
    color: {C["muted"]} !important;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 8px 20px;
    transition: all 0.2s;
    border: none !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {C["ink"]} !important;
    background: {C["primary"]}15;
}}
.stTabs [aria-selected="true"] {{
    background: {C["gradient"]} !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px -4px {C["primary"]}80;
}}
.stTabs [data-baseweb="tab-highlight"] {{
    display: none;
}}
.stTabs [data-baseweb="tab-border"] {{
    display: none;
}}

/* ============= SECTION TITLE ============= */
.section-title {{
    font-family: 'Sora', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: {C["ink"]} !important;
    margin: 1.5rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.section-title::before {{
    content: '';
    width: 4px;
    height: 20px;
    background: {C["gradient"]};
    border-radius: 2px;
}}

/* ============= DATAFRAME ============= */
[data-testid="stDataFrame"] {{
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {C["border"]};
}}

/* ============= EXPANDER ============= */
[data-testid="stExpander"] {{
    background: {C["card"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
}}
[data-testid="stExpander"] summary {{
    color: {C["ink"]} !important;
    font-weight: 600;
}}

/* ============= ALERTS ============= */
[data-testid="stAlert"] {{
    border-radius: 12px;
}}

/* ============= SCROLLBAR ============= */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}
::-webkit-scrollbar-track {{
    background: {C["bg"]};
}}
::-webkit-scrollbar-thumb {{
    background: {C["border"]};
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {C["muted"]};
}}
</style>
""", unsafe_allow_html=True)


# =======================================================================
# HELPER FUNCTIONS
# =======================================================================
def stat_card(value, label, accent=None, icon=None):
    accent = accent or C["primary"]
    icon_html = f'<div class="stat-icon">{icon}</div>' if icon else ""
    st.markdown(
        f"""<div class="stat-card" style="--accent: {accent};">
                {icon_html}
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


# =======================================================================
# DATA LOADING
# =======================================================================
def clean_gender(gender):
    if pd.isna(gender):
        return "Unknown"
    gender = str(gender).lower().strip()
    male_terms = ["male", "m", "man", "cis male", "mal", "male-ish", "maile",
                  "malr", "cis man", "make", "guy (-ish) ^_^", "male leaning androgynous"]
    female_terms = ["female", "f", "woman", "cis female", "femake", "female ",
                    "cis-female/femme", "female (cis)", "femail"]
    trans_terms = ["trans-female", "trans woman", "female (trans)"]
    if gender in male_terms or ("male" in gender and "female" not in gender and "trans" not in gender):
        return "Male"
    elif gender in female_terms or ("female" in gender and "trans" not in gender and "male" not in gender):
        return "Female"
    elif gender in trans_terms or "trans" in gender:
        return "Trans"
    return "Other"


@st.cache_data
def load_data(path="survey.csv"):
    df = pd.read_csv(path)
    df_clean = df.copy()
    df_clean = df_clean[(df_clean["Age"] >= 15) & (df_clean["Age"] <= 75)]
    df_clean["Gender"] = df_clean["Gender"].apply(clean_gender)
    df_clean["self_employed"] = df_clean["self_employed"].fillna("No")
    df_clean["work_interfere"] = df_clean["work_interfere"].fillna("Don't know")
    if "comments" in df_clean.columns:
        df_clean = df_clean.drop(columns=["comments"])
    if "state" in df_clean.columns:
        df_clean["state"] = df_clean["state"].fillna("N/A")
    bins = [15, 25, 35, 45, 55, 75]
    labels = ["15-24", "25-34", "35-44", "45-54", "55+"]
    df_clean["Age_Group"] = pd.cut(df_clean["Age"], bins=bins, labels=labels)
    return df_clean


try:
    df_clean = load_data()
except FileNotFoundError:
    st.error("❌ **survey.csv not found!** Place it in the same folder as `app.py`.")
    st.stop()

# =======================================================================
# SIDEBAR
# =======================================================================
with st.sidebar:
    st.title("🧠 Mental Health Survey")
    st.caption("OSMI · 2014 Dataset")

    # Dark mode toggle
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown("**Appearance**")
    with col_b:
        dark = st.toggle("🌙", value=st.session_state.dark_mode, key="dark_toggle")
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

    st.markdown("---")
    st.markdown("### 🎛️ Filters")
    st.caption("Narrow the dataset to explore specific segments.")

    age_min, age_max = int(df_clean["Age"].min()), int(df_clean["Age"].max())
    age_range = st.slider("Age range", age_min, age_max, (age_min, age_max))

    genders = sorted(df_clean["Gender"].unique().tolist())
    selected_genders = st.multiselect("Gender", genders, default=genders)

    top_countries_all = df_clean["Country"].value_counts().index.tolist()
    selected_countries = st.multiselect("Country (empty = all)", top_countries_all, default=[])

    company_sizes = df_clean["no_employees"].dropna().unique().tolist()
    selected_sizes = st.multiselect("Company size", company_sizes, default=company_sizes)

    treatment_filter = st.radio("Sought treatment?", ["All", "Yes", "No"], index=0, horizontal=True)
    remote_only = st.checkbox("Remote workers only")

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

# Sidebar stat
with st.sidebar:
    st.markdown(
        f"""<div class="sidebar-stat">
                <div class="sidebar-stat-value">{len(data):,}</div>
                <div class="sidebar-stat-label">of {len(df_clean):,} respondents</div>
            </div>""",
        unsafe_allow_html=True,
    )

if data.empty:
    st.warning("⚠️ No respondents match the current filters. Try widening them.")
    st.stop()

# =======================================================================
# HERO BANNER
# =======================================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-eyebrow">OSMI Survey · 2014 · Tech Industry</div>
        <div style="display:flex; align-items:flex-end; justify-content:space-between; gap:2rem; flex-wrap:wrap; position:relative; z-index:1;">
            <div style="flex:1; min-width:300px;">
                <div class="hero-title">Mental Health in Tech</div>
                <div class="hero-sub">
                    How tech workers experience, disclose, and get support for mental health
                    at work — explored across <b>{len(df_clean):,}</b> survey responses.
                </div>
            </div>
            <div style="text-align:right;">
                <div class="hero-number">{len(data):,}</div>
                <div class="hero-number-label">In View</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =======================================================================
# STAT CARDS
# =======================================================================
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    stat_card(f"{data['Age'].mean():.1f}", "Average age", C["primary"], "👤")
with c2:
    stat_card(f"{(data['treatment'] == 'Yes').mean() * 100:.0f}%", "Sought treatment", C["accent"], "💚")
with c3:
    stat_card(f"{(data['family_history'] == 'Yes').mean() * 100:.0f}%", "Family history", C["purple"], "🧬")
with c4:
    stat_card(f"{(data['remote_work'] == 'Yes').mean() * 100:.0f}%", "Remote workers", C["warning"], "🏠")
with c5:
    stat_card(f"{(data['benefits'] == 'Yes').mean() * 100:.0f}%", "Have benefits", C["success"], "🎁")

st.write("")
st.write("")

# =======================================================================
# TABS
# =======================================================================
tab_overview, tab_uni, tab_bi, tab_multi, tab_insights = st.tabs(
    ["📊 Overview", "📈 Univariate", "🔀 Bivariate", "🌐 Multivariate", "💡 Insights"]
)

# ==================== OVERVIEW ====================
with tab_overview:
    section_title("📋 Dataset Preview")
    st.dataframe(data.head(50), use_container_width=True, height=400)

    with st.expander("📊 Column Summary & Statistics"):
        st.dataframe(data.describe(include="all").transpose(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📌 Quick Facts")
        st.markdown(f"""
        - **Total respondents in view:** {len(data):,}
        - **Top country:** {data['Country'].value_counts().index[0] if len(data) else 'N/A'}
        - **Age range:** {data['Age'].min():.0f} – {data['Age'].max():.0f} years
        - **Most common age group:** {data['Age_Group'].mode()[0] if len(data) else 'N/A'}
        """)
    with col2:
        st.markdown("### 🏢 Company Insights")
        st.markdown(f"""
        - **Tech company workers:** {(data['tech_company'] == 'Yes').mean() * 100:.1f}%
        - **Self-employed:** {(data['self_employed'] == 'Yes').mean() * 100:.1f}%
        - **Remote workers:** {(data['remote_work'] == 'Yes').mean() * 100:.1f}%
        - **Most common company size:** {data['no_employees'].mode()[0] if len(data) else 'N/A'}
        """)

# ==================== UNIVARIATE ====================
with tab_uni:
    section_title("📊 Single-Variable Distributions")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(data, x="Age", nbins=30,
                          title="Age Distribution",
                          color_discrete_sequence=[C["primary"]])
        fig.update_layout(xaxis_title="Age", yaxis_title="Count", bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        gender_counts = data["Gender"].value_counts()
        fig = px.pie(values=gender_counts.values, names=gender_counts.index,
                    title="Gender Distribution", hole=0.55,
                    color=gender_counts.index, color_discrete_map=GENDER_MAP)
        fig.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        country_counts = data["Country"].value_counts().head(15)
        fig = px.bar(x=country_counts.values, y=country_counts.index, orientation="h",
                    title="Top 15 Countries",
                    color=country_counts.values,
                    color_continuous_scale=[C["primary"] + "40", C["primary"]])
        fig.update_layout(xaxis_title="Count", yaxis_title="",
                         yaxis=dict(autorange="reversed"),
                         coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.histogram(data, x="no_employees",
                          title="Company Size Distribution",
                          color_discrete_sequence=[C["purple"]])
        fig.update_layout(xaxis_title="Number of employees", yaxis_title="Count", bargap=0.15)
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        treatment_counts = data["treatment"].value_counts()
        fig = px.pie(values=treatment_counts.values, names=treatment_counts.index,
                    title="Sought Treatment?", hole=0.55,
                    color=treatment_counts.index, color_discrete_map=SENTIMENT_MAP)
        fig.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        fh_counts = data["family_history"].value_counts()
        fig = px.bar(x=fh_counts.index, y=fh_counts.values,
                    title="Family History of Mental Illness",
                    color=fh_counts.index, color_discrete_map=SENTIMENT_MAP)
        fig.update_layout(xaxis_title="", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ==================== BIVARIATE ====================
with tab_bi:
    section_title("🔀 Two-Variable Relationships")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(data, x="Gender", color="treatment", barmode="group",
                          title="Treatment Seeking by Gender",
                          color_discrete_map=SENTIMENT_MAP)
        fig.update_layout(xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(data, x="family_history", color="treatment", barmode="group",
                          title="Treatment by Family History",
                          color_discrete_map=SENTIMENT_MAP)
        fig.update_layout(xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(data, x="work_interfere", color="treatment", barmode="group",
                          title="Work Interference by Treatment",
                          category_orders={"work_interfere": WORK_INTERFERE_ORDER},
                          color_discrete_map=SENTIMENT_MAP)
        fig.update_layout(xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.histogram(data, x="no_employees", color="benefits", barmode="group",
                          title="Benefits by Company Size",
                          color_discrete_map=SENTIMENT_MAP)
        fig.update_layout(xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        fig = px.histogram(data, x="mental_health_consequence", color="Gender",
                          barmode="group",
                          title="Mental Health Consequences by Gender",
                          color_discrete_map=GENDER_MAP)
        fig.update_layout(xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        fig = px.histogram(data, x="supervisor", color="treatment", barmode="group",
                          title="Willingness to Discuss with Supervisor",
                          color_discrete_map=SENTIMENT_MAP)
        fig.update_layout(xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

# ==================== MULTIVARIATE ====================
with tab_multi:
    section_title("🌐 Multi-Variable Analysis")

    top5 = data["Country"].value_counts().head(5).index
    df_top = data[data["Country"].isin(top5)]

    if not df_top.empty:
        treatment_by_country_gender = (
            df_top.groupby(["Country", "Gender"])["treatment"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Treatment_Rate")
        )
        fig = px.bar(treatment_by_country_gender, x="Country", y="Treatment_Rate",
                    color="Gender", barmode="group",
                    title="Treatment Rate by Country & Gender (Top 5 Countries)",
                    color_discrete_map=GENDER_MAP)
        fig.update_layout(yaxis_title="Treatment rate (%)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        cross_tab = pd.crosstab(data["benefits"], data["care_options"])
        fig = px.imshow(cross_tab, text_auto=True, aspect="auto",
                       title="Benefits vs. Care Options",
                       color_continuous_scale=[C["primary"] + "30", C["primary_dark"]])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.sunburst(data, path=["no_employees", "benefits", "treatment"],
                         title="Company Size → Benefits → Treatment",
                         color="treatment", color_discrete_map=SENTIMENT_MAP)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(data, x="treatment", y="Age", color="Gender",
                    title="Age Distribution by Treatment & Gender",
                    color_discrete_map=GENDER_MAP)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.violin(data, x="work_interfere", y="Age", color="treatment", box=True,
                       title="Age by Work Interference & Treatment",
                       category_orders={"work_interfere": WORK_INTERFERE_ORDER},
                       color_discrete_map=SENTIMENT_MAP)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    section_title("🔥 Correlation Heatmap")
    df_encoded = data.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include=["object"]).columns:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    corr_cols = ["Age", "family_history", "treatment", "work_interfere", "benefits",
                 "care_options", "wellness_program", "seek_help", "anonymity",
                 "mental_health_consequence", "phys_health_consequence", "coworkers",
                 "supervisor", "mental_health_interview", "mental_vs_physical",
                 "obs_consequence"]
    corr_cols = [c for c in corr_cols if c in df_encoded.columns]
    corr_matrix = df_encoded[corr_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                    color_continuous_scale=[C["accent"], C["card"], C["primary_dark"]],
                    zmin=-1, zmax=1)
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    # Mental vs Physical
    section_title("🧠 Mental vs Physical Health Perception")
    mental_vs_physical = (
        pd.crosstab(data["no_employees"], data["mental_vs_physical"],
                    normalize="index") * 100
    ).reset_index().melt(id_vars="no_employees", var_name="mental_vs_physical",
                          value_name="Percentage")
    fig = px.bar(mental_vs_physical, x="no_employees", y="Percentage",
                color="mental_vs_physical", barmode="stack",
                title="Mental vs. Physical Health Perception by Company Size (%)",
                color_discrete_map=SENTIMENT_MAP)
    fig.update_layout(xaxis_title="Company size", yaxis_title="Percentage (%)")
    st.plotly_chart(fig, use_container_width=True)

# ==================== INSIGHTS ====================
with tab_insights:
    section_title("💡 Key Insights & Recommendations")

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        stat_card(f"{len(data):,}", "Total respondents", C["primary"], "👥")
        st.write("")
        stat_card(f"{data['Age'].mean():.1f} yrs", "Average age", C["purple"], "🎂")
    with ic2:
        stat_card(f"{(data['treatment'] == 'Yes').mean() * 100:.1f}%",
                  "Treatment rate", C["accent"], "💚")
        st.write("")
        stat_card(f"{(data['family_history'] == 'Yes').mean() * 100:.1f}%",
                  "Family history rate", C["success"], "🧬")
    with ic3:
        stat_card(f"{(data['self_employed'] == 'Yes').mean() * 100:.1f}%",
                  "Self-employed", C["warning"], "💼")
        st.write("")
        stat_card(f"{(data['obs_consequence'] == 'Yes').mean() * 100:.1f}%",
                  "Observed neg. consequences", C["pink"], "⚠️")

    st.write("")
    section_title("📌 Summary")

    st.markdown(f"""
    - **Top country:** {data['Country'].value_counts().index[0] if len(data) else 'N/A'}
    - **Gender split:** {', '.join(f"{k}: {v}" for k, v in data['Gender'].value_counts().to_dict().items())}
    - **Remote workers:** {(data['remote_work'] == 'Yes').mean() * 100:.1f}%
    - **Tech company workers:** {(data['tech_company'] == 'Yes').mean() * 100:.1f}%
    """)

    st.write("")
    section_title("🎯 Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏢 For Employers")
        st.markdown("""
        1. **Normalize mental health conversations** — leadership must model openness
        2. **Communicate benefits clearly** — many employees don't know what's available
        3. **Train managers** to recognize and support mental health challenges
        4. **Create anonymous support channels** for those who fear stigma
        5. **Offer flexible work** to help employees manage their conditions
        """)

    with col2:
        st.markdown("#### 👤 For Employees")
        st.markdown("""
        1. **Know your resources** — check your benefits, EAP, and coverage
        2. **Build a trusted network** — 1-2 colleagues who can support you
        3. **Seek help early** — don't wait for a crisis
        4. **Document your needs** — help HR understand required accommodations
        5. **Advocate for change** — share your story (if safe) to reduce stigma
        """)

# =======================================================================
# FOOTER
# =======================================================================
st.markdown("---")
st.markdown(
    f"""<div style='text-align: center; color: {C["muted"]}; padding: 1rem 0;'>
    Built with ❤️ using Streamlit · Data: OSMI Mental Health in Tech Survey (2014)
    </div>""",
    unsafe_allow_html=True,
)
