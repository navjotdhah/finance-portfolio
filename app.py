import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Finance Portfolio Dashboard", layout="wide")

st.title("Interactive Finance Dashboard: DCF + Comparables + Option Greeks")

# ------------------------
# Sample Financial Data
# ------------------------
default_wacc = 0.09
default_growth = 0.05
default_terminal_growth = 0.025

financials = {
    "Income Statement": pd.DataFrame({
        "Year": ["2022","2023","2024","2025"],
        "Revenue": [1000,1200,1400,1600],
        "EBIT": [200,240,280,320],
        "Net Income": [150,180,210,240]
    }),
    "Balance Sheet": pd.DataFrame({
        "Item": ["Cash","Debt","Equity"],
        "2022":[100,50,200],
        "2023":[120,60,220],
        "2024":[140,70,240],
        "2025":[160,80,260]
    }),
    "Cash Flow": pd.DataFrame({
        "Year": ["2022","2023","2024","2025"],
        "Operating CF": [180,210,250,280],
        "Investing CF": [-50,-60,-70,-80],
        "Financing CF": [20,30,40,50]
    })
}

# Default comparables
comparables = ["AAPL", "MSFT", "GOOGL"]

# ------------------------
# Sidebar Inputs
# ------------------------
st.sidebar.header("DCF & Comparables Settings")

discount = st.sidebar.slider("Discount rate / WACC (%)", 0.0, 30.0, float(default_wacc*100), 0.1) / 100
growth = st.sidebar.slider("Projection growth rate (%)", 0.0, 30.0, float(default_growth*100), 0.1) / 100
terminal_growth = st.sidebar.slider("Terminal growth (%)", 0.0, 10.0, float(default_terminal_growth*100), 0.1) / 100
projection_years = st.sidebar.number_input("Projection horizon (years)", min_value=1, max_value=10, value=5, step=1)
user_comps = st.sidebar.multiselect("Select comparables", options=comparables, default=comparables)

# ------------------------
# Financial Statements
# ------------------------
st.header("Financial Statements")
for name, df in financials.items():
    st.subheader(name)
    st.dataframe(df, use_container_width=True) if not df.empty else st.info("N/A")

# ------------------------
# DCF Calculation
# ------------------------
st.header("Discounted Cash Flow (DCF) Analysis")

last_cf = financials["Cash Flow"]["Operating CF"].iloc[-1]
years = np.arange(1, projection_years + 1)
projected_cf = [last_cf * (1+growth)**i for i in years]
discounted_cf = [cf / ((1+discount)**i) for i, cf in enumerate(projected_cf, 1)]
dcf_value = sum(discounted_cf)

terminal_value = projected_cf[-1] * (1 + terminal_growth) / (discount - terminal_growth)
terminal_value_pv = terminal_value / ((1+discount)**projection_years)
total_value = dcf_value + terminal_value_pv

df_dcf = pd.DataFrame({
    "Year": [f"Year {i}" for i in years] + ["Terminal"],
    "Projected CF": projected_cf + [terminal_value],
    "Discounted CF": discounted_cf + [terminal_value_pv]
})

st.dataframe(df_dcf, use_container_width=True)

fig_dcf = go.Figure()
fig_dcf.add_trace(go.Bar(x=df_dcf["Year"], y=df_dcf["Discounted CF"], name="Discounted CF"))
fig_dcf.add_trace(go.Scatter(x=df_dcf["Year"], y=df_dcf["Projected CF"], mode="lines+markers", name="Projected CF"))
st.plotly_chart(fig_dcf, use_container_width=True)

st.metric("DCF Valuation (PV + Terminal)", f"${total_value:,.2f}")

# ------------------------
# Comparables Analysis
# ------------------------
st.header("Comparables Analysis")
st.write(f"Showing selected comparables: {', '.join(user_comps)}")

comp_data = pd.DataFrame({
    "Company": user_comps,
    "P/E": [25, 30, 28],
    "EV/EBITDA": [15, 18, 17],
    "P/B": [5, 6, 4]
})
st.dataframe(comp_data, use_container_width=True)

# ------------------------
# Option Greeks Calculator
# ------------------------
st.header("Option Greeks Calculator")

with st.expander("Show/Hide Option Greeks Inputs"):
    S = st.number_input("Underlying Price (S)", value=100.0)
    K = st.number_input("Strike Price (K)", value=100.0)
    T = st.number_input("Time to Maturity (Years)", value=1.0)
    r = st.number_input("Risk-Free Rate (%)", value=5.0)/100
    sigma = st.number_input("Volatility (%)", value=20.0)/100
    option_type = st.selectbox("Option Type", ["Call","Put"])

def black_scholes_price(S, K, T, r, sigma, option_type="Call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == "Call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def greeks(S, K, T, r, sigma, option_type="Call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = norm.cdf(d1) if option_type=="Call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*(norm.cdf(d2) if option_type=="Call" else norm.cdf(-d2))
    vega = S*norm.pdf(d1)*np.sqrt(T)
    rho = K*T*np.exp(-r*T)*(norm.cdf(d2) if option_type=="Call" else -norm.cdf(-d2))
    return delta, gamma, theta, vega, rho

price = black_scholes_price(S,K,T,r,sigma,option_type)
delta, gamma, theta, vega, rho = greeks(S,K,T,r,sigma,option_type)

st.subheader(f"{option_type} Option Price: ${price:,.2f}")
st.write(f"Delta: {delta:.4f}")
st.write(f"Gamma: {gamma:.4f}")
st.write(f"Theta: {theta:.4f}")
st.write(f"Vega: {vega:.4f}")
st.write(f"Rho: {rho:.4f}")
