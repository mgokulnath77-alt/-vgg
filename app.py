import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import pearsonr, ttest_ind

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Biomarker Discovery Lab", layout="wide")

# --- GLASSMORPHISM STYLE ---
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); }
    .stMetric { background: rgba(255, 255, 255, 0.4); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.2); }
    div.stPlotlyChart { background: rgba(255, 255, 255, 0.3); backdrop-filter: blur(10px); border-radius: 15px; border: 1px solid rgba(255,255,255,0.2); }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def generate_biomarker_data():
    np.random.seed(42)
    n = 200
    # Simulating two genes with a degree of co-expression
    base_signal = np.random.normal(10, 2, n)
    gene_a = base_signal + np.random.normal(0, 1, n)
    gene_b = base_signal * 0.5 + np.random.normal(5, 1.5, n)
    
    # Clinical Outcomes
    # If Gene A is high AND Gene B is high, survival is lower (Synergy)
    risk = (gene_a * 0.4) + (gene_b * 0.6)
    survival_time = np.random.exponential(100 / (risk/10))
    event = np.random.choice([1, 0], size=n, p=[0.8, 0.2])
    
    return pd.DataFrame({
        'Sample_ID': [f'PATIENT_{i:03d}' for i in range(n)],
        'Group': ['Tumor']*100 + ['Normal']*100,
        'Gene_A': gene_a,
        'Gene_B': gene_b,
        'Survival_Months': survival_time,
        'Status': event
    })

df = generate_biomarker_data()

# --- SIDEBAR ---
st.sidebar.title("🧬 Biomarker Finder")
st.sidebar.markdown("Comparing **Gene A** vs **Gene B**")

menu = st.sidebar.selectbox("Analysis Step", 
    ["1. Co-Expression Analysis", "2. Ratio-Based Biomarkers", "3. Survival Synergy", "4. Data Explorer"])

# --- MODULE 1: CO-EXPRESSION SCATTER ---
if menu == "1. Co-Expression Analysis":
    st.title("📈 Co-Expression Scatter Plot")
    st.write("Determining the regulatory relationship between the two genes.")

    

    corr, p_val = pearsonr(df['Gene_A'], df['Gene_B'])
    
    fig = px.scatter(df, x='Gene_A', y='Gene_B', color='Group', 
                     trendline="ols", template="plotly_white",
                     marginal_x="histogram", marginal_y="box")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Pearson R", f"{corr:.3f}")
    c2.metric("P-Value", f"{p_val:.2e}")
    c3.metric("Relationship", "Positive" if corr > 0 else "Negative")
    
    st.plotly_chart(fig, use_container_width=True)

# --- MODULE 2: RATIO-BASED BIOMARKERS ---
elif menu == "2. Ratio-Based Biomarkers":
    st.title("⚖️ Ratio-Based Biomarker Analysis")
    st.write("Analyzing the Gene A / Gene B ratio as a potential diagnostic indicator.")
    
    df['A_B_Ratio'] = df['Gene_A'] / df['Gene_B']
    
    fig = px.violin(df, x='Group', y='A_B_Ratio', color='Group', box=True, points="all",
                    template="plotly_white", color_discrete_sequence=['#ff4b4b', '#1f77b4'])
    
    # Stats
    t_res = ttest_ind(df[df['Group']=='Tumor']['A_B_Ratio'], df[df['Group']=='Normal']['A_B_Ratio'])
    
    st.metric("Ratio Significance (P-Value)", f"{t_res.pvalue:.4e}")
    st.plotly_chart(fig, use_container_width=True)

# --- MODULE 3: SURVIVAL SYNERGY ---
elif menu == "3. Survival Synergy":
    st.title("⚔️ Survival Synergy (Combinatorial)")
    st.write("Categorizing patients into 4 quadrants based on Median Expression of both genes.")
    
    # Calculate Medians
    med_a = df['Gene_A'].median()
    med_b = df['Gene_B'].median()
    
    def categorize(row):
        a = "High A" if row['Gene_A'] > med_a else "Low A"
        b = "High B" if row['Gene_B'] > med_b else "Low B"
        return f"{a} / {b}"

    df['Synergy_Group'] = df.apply(categorize, axis=1)
    
    
    
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    
    for group in df['Synergy_Group'].unique():
        g_data = df[df['Synergy_Group'] == group]
        kmf.fit(g_data['Survival_Months'], g_data['Status'], label=group)
        fig.add_trace(go.Scatter(x=kmf.timeline, y=kmf.survival_function_.iloc[:, 0], 
                                 name=group, mode='lines', line_shape='hv'))
        
    fig.update_layout(template="plotly_white", title="Dual-Gene Stratified Survival",
                      xaxis_title="Months", yaxis_title="Survival Probability")
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("Groups showing the steepest drop represent the highest-risk biomarkers.")

# --- MODULE 4: DATA EXPLORER ---
elif menu == "4. Data Explorer":
    st.title("🔍 Raw Biomarker Data")
    st.dataframe(df.style.background_gradient(cmap='Blues'), use_container_width=True)