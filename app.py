# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import math

# --- Safe import of norm (scipy optional) ---
try:
    from scipy.stats import norm
except Exception:
    class _NormFallback:
        @staticmethod
        def cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        @staticmethod
        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)
    norm = _NormFallback()

# -------------------------
# Page config & CSS
# -------------------------
st.set_page_config(page_title="Analyst Terminal — Valuation & Options", page_icon="💹", layout="wide")

st.markdown("""
<style>
body { background-color: #0e1117; color: #e6e6e6; }
h1,h2,h3 { color: #39FF14; font-weight:700; }
.block-container { padding: 1rem 2rem; }
.metric-card { background: #111316; padding: 10px; border-radius: 8px; border: 1px solid #222; }
.css-1d391kg { background-color: #0b0c0e !important; }
.stDataFrame { background-color: #121416; }
a { color: #7ef9a4; }
</style>
""", unsafe_allow_html=True)

st.title("💹 Analyst Terminal — Equity Valuation & Options")
st.caption('Real-time modelling, DCF, comps, Black–Scholes options, and news. Built for IB/PE/AM prep. — [Navjot Dhah](https://www.linkedin.com/in/navjot-dhah-57870b238)')

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Search & Settings")
ticker = st.sidebar.text_input("Enter ticker (example: AAPL, WYNN, MSFT)", value="WYNN").upper().strip()
use_live = st.sidebar.checkbox("Use live yfinance data", value=True)

# DCF defaults
default_wacc = st.sidebar.number_input("Default WACC (%)", value=9.0, step=0.1)/100.0
default_tg = st.sidebar.number_input("Default Terminal growth (%)", value=2.5, step=0.1)/100.0
projection_years = st.sidebar.selectbox("Projection years", [5,7,10], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: Override any value if yfinance misses data.")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(ttl=300)
def fetch_yf(t):
    tk = yf.Ticker(t)
    try: info = tk.info
    except: info = {}
    try: fin = tk.financials
    except: fin = pd.DataFrame()
    try: bs = tk.balance_sheet
    except: bs = pd.DataFrame()
    try: cf = tk.cashflow
    except: cf = pd.DataFrame()
    try: hist = tk.history(period="5y")
    except: hist = pd.DataFrame()
    return info, fin, bs, cf, hist

def safe_number(x):
    try: return float(x)
    except: return np.nan

def style_numeric(df):
    if df is None or df.empty: return df
    df_t = df.T
    numeric_cols = df_t.select_dtypes(include=[np.number]).columns
    fmt = {c: "{:,.0f}" for c in numeric_cols}
    return df_t.style.format(fmt)

def find_row_value(df, keywords):
    if df is None or df.empty: return None
    idx = df.index
    for k in keywords:
        for label in idx:
            if k.lower() in str(label).lower():
                try: return df.loc[label].iloc[0]
                except: continue
    return None

def dcf_from_fcf(last_fcf, growth, discount, tg, years):
    proj = [last_fcf * (1 + growth)**i for i in range(1, years+1)]
    pv = sum([proj[i] / ((1 + discount)**(i+1)) for i in range(len(proj))])
    if discount <= tg: terminal = np.nan
    else:
        terminal_nom = proj[-1] * (1 + tg) / (discount - tg)
        terminal = terminal_nom / ((1 + discount)**years)
    enterprise = pv + (terminal if not np.isnan(terminal) else 0)
    return {
        "proj_nominal": proj,
        "proj_pv": [proj[i] / ((1 + discount)**(i+1)) for i in range(len(proj))],
        "terminal_pv": terminal,
        "enterprise_value": enterprise
    }

def black_scholes_price(S, K, T, r, sigma, option="call"):
    if T <= 0 or sigma <= 0:
        return max(0.0, S-K) if option=="call" else max(0.0, K-S)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option=="call":
        return S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
    else:
        return K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_scholes_greeks(S,K,T,r,sigma,option="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = norm.cdf(d1) if option=="call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T)/100
    theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*math.exp(-r*T)*(norm.cdf(d2) if option=="call" else norm.cdf(-d2)))/365
    rho = (K*T*math.exp(-r*T)* (norm.cdf(d2) if option=="call" else -norm.cdf(-d2)))/100
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

def get_yahoo_news(ticker, limit=6):
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
        resp = requests.get(url, timeout=6).json()
        items = resp.get("news",[]) or resp.get("items",[]) or []
        out=[]
        for it in items[:limit]:
            title = it.get("title") or it.get("headline")
            link = it.get("link") or it.get("url")
            pub = it.get("publisher") or it.get("provider") or it.get("source")
            ts = it.get("providerPublishTime") or it.get("pubDate")
            time = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M") if ts and isinstance(ts,(int,float)) else ""
            out.append({"title": title,"link": link,"source":pub,"time":time})
        return out
    except: return []

# -------------------------
# Fetch & display
# -------------------------
info, fin, bs, cf, hist = fetch_yf(ticker) if use_live else ({}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
company_name = info.get("shortName") or info.get("longName") or ticker
st.header(f"{company_name} — {ticker}")

# Metrics
price = safe_number(info.get("currentPrice") or (hist["Close"].iloc[-1] if not hist.empty else np.nan))
market_cap = safe_number(info.get("marketCap"))
shares_out = safe_number(info.get("sharesOutstanding") or info.get("floatShares") or np.nan)
ev = (market_cap or 0) + safe_number(info.get("totalDebt") or 0) - safe_number(info.get("totalCash") or 0)

col1,col2,col3,col4,col5 = st.columns(5)
col1.metric("Price", f"${price:,.2f}" if not np.isnan(price) else "N/A")
col2.metric("Market Cap", f"${market_cap:,.0f}" if not np.isnan(market_cap) else "N/A")
col3.metric("Enterprise Value", f"${ev:,.0f}" if not np.isnan(ev) else "N/A")
col4.metric("Shares Out.", f"{shares_out:,.0f}" if not np.isnan(shares_out) else "N/A")
col5.metric("Sector", info.get("sector","N/A"))

# -------------------------
# Financial Statements
# -------------------------
f1,f2,f3 = st.tabs(["Income Statement","Balance Sheet","Cash Flow"])
with f1:
    st.markdown("**Income Statement**")
    if not fin.empty:
        st.dataframe(style_numeric(fin), use_container_width=True)
    else:
        st.info("N/A")
with f2:
    st.markdown("**Balance Sheet**")
    if not bs.empty:
        st.dataframe(style_numeric(bs), use_container_width=True)
    else:
        st.info("N/A")
with f3:
    st.markdown("**Cash Flow**")
    if not cf.empty:
        st.dataframe(style_numeric(cf), use_container_width=True)
    else:
        st.info("N/A")

# -------------------------
# DCF
# -------------------------
st.subheader("📊 Discounted Cash Flow (DCF) Model")
last_fcf = find_row_value(cf, ["Free Cash Flow", "Total Cash From Operating Activities"]) or 1000
growth = st.slider("Projection growth rate (%)", 0, 30, 5)/100
discount = st.slider("Discount rate / WACC (%)", 0,30,float(default_wacc*100))/100
tg = st.slider("Terminal growth (%)", 0,10,float(default_tg*100))/100

dcf_result = dcf_from_fcf(last_fcf, growth, discount, tg, projection_years)
proj_df = pd.DataFrame({
    "Year": [i+1 for i in range(projection_years)],
    "FCF": dcf_result["proj_nominal"],
    "Discounted FCF": dcf_result["proj_pv"]
})
st.dataframe(proj_df, use_container_width=True)
st.markdown(f"**Terminal PV:** ${dcf_result['terminal_pv']:,.0f}  \n**Enterprise Value:** ${dcf_result['enterprise_value']:,.0f}")

# -------------------------
# Comparables
# -------------------------
st.subheader("📈 Comparable Companies")
default_comps = ["WYNN","MGM","CZR"]
user_comps = st.text_input("Enter comparables (comma separated)", value=",".join(default_comps)).upper().split(",")
comp_data=[]
for t in user_comps:
    i,f,b,c,h = fetch_yf(t.strip())
    comp_price = safe_number(i.get("currentPrice"))
    comp_mc = safe_number(i.get("marketCap"))
    comp_ev = (comp_mc or 0) + safe_number(i.get("totalDebt") or 0) - safe_number(i.get("totalCash") or 0)
    comp_data.append({"Ticker":t.strip(),"Price":comp_price,"Market Cap":comp_mc,"EV":comp_ev})
st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

# -------------------------
# Options / Greeks
# -------------------------
st.subheader("📈 Black-Scholes Options Calculator")
S = st.number_input("Underlying price (S)", value=float(price) if not np.isnan(price) else 100.0)
K = st.number_input("Strike price (K)", value=S)
T = st.number_input("Time to expiration (years)", value=0.25)
r = st.number_input("Risk-free rate (%)", value=5.0)/100
sigma = st.number_input("Volatility (%)", value=30.0)/100
option_type = st.selectbox("Option Type", ["call","put"])

bs_price = black_scholes_price(S,K,T,r,sigma,option_type)
bs_greeks = black_scholes_greeks(S,K,T,r,sigma,option_type)
st.metric("Option Price", f"${bs_price:,.2f}")
st.dataframe(pd.DataFrame([bs_greeks]))

# -------------------------
# Price Chart
# -------------------------
st.subheader("📉 Price Chart")
if not hist.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Close"))
    fig.update_layout(title=f"{ticker} Price History", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No historical data available")

# -------------------------
# News
# -------------------------
st.subheader("📰 Recent News")
news_items = get_yahoo_news(ticker)
for n in news_items:
    st.markdown(f"- [{n['title']}]({n['link']}) — *{n['source']}* {n['time']}")

st.markdown("---")
st.caption("Built with Streamlit + yfinance + Plotly. Shows DCF, comparables, options pricing, financials, and news.")
