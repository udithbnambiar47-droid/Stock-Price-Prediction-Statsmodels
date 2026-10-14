import warnings
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock OLS Regression", page_icon="📈", layout="wide")
st.title("📈 Stock Price OLS Regression")
st.caption("Download price data from Yahoo Finance and fit an OLS model with statsmodels.")

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Downloading data from Yahoo Finance...", ttl=3600)
def load_data(ticker: str, years: int) -> pd.DataFrame:
    end = date.today() + timedelta(days=1)  # yfinance `end` is exclusive
    start = end - timedelta(days=int(years * 365))
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    # Newer yfinance returns MultiIndex columns like ('Close', 'GOOGL') -> flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = [c for c in PRICE_COLS if c in df.columns]
    return df[cols].dropna()


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", value="GOOGL").strip().upper()
    years = st.slider("Years of history", min_value=1, max_value=15, value=5)

    target = st.selectbox("Dependent variable (y)", PRICE_COLS, index=3)
    default_x = [c for c in ["Open", "Volume"] if c != target]
    features = st.multiselect(
        "Independent variables (X)",
        [c for c in PRICE_COLS if c != target],
        default=default_x,
    )
    run = st.button("Run regression", type="primary", use_container_width=True)

if not ticker:
    st.info("Enter a ticker symbol in the sidebar.")
    st.stop()

data = load_data(ticker, years)

if data.empty:
    st.error(f"No data returned for **{ticker}**. Check the ticker symbol and try again.")
    st.stop()

# --------------------------------------------------------------------------
# Data overview
# --------------------------------------------------------------------------
st.subheader(f"{ticker} — data overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(data):,}")
c2.metric("From", data.index.min().strftime("%Y-%m-%d"))
c3.metric("To", data.index.max().strftime("%Y-%m-%d"))

with st.expander("Preview data", expanded=True):
    st.dataframe(data.head(), use_container_width=True)

# --------------------------------------------------------------------------
# Correlation matrix
# --------------------------------------------------------------------------
st.subheader("Correlation matrix")
corr = data.corr()
col_a, col_b = st.columns([1, 1])
with col_a:
    st.dataframe(corr.style.format("{:.4f}"), use_container_width=True)
with col_b:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------------------------------
# Pair plot
# --------------------------------------------------------------------------
st.subheader("Pair plot with regression lines")
pair_cols = [target] + features
if len(pair_cols) >= 2:
    if st.checkbox("Show pair plot (can take a few seconds)", value=True):
        g = sns.pairplot(
            data[pair_cols],
            kind="reg",
            plot_kws={"line_kws": {"color": "black"}, "scatter_kws": {"s": 8, "alpha": 0.5}},
        )
        st.pyplot(g.figure)
        plt.close(g.figure)
else:
    st.info("Select at least one independent variable to see the pair plot.")

# --------------------------------------------------------------------------
# OLS regression
# --------------------------------------------------------------------------
st.subheader("OLS regression")

if not features:
    st.warning("Select at least one independent variable in the sidebar.")
    st.stop()

if not run:
    st.info("Click **Run regression** in the sidebar to fit the model.")
    st.stop()

y = data[target]
X = sm.add_constant(data[features])
result = sm.OLS(y, X).fit()

m1, m2, m3, m4 = st.columns(4)
m1.metric("R²", f"{result.rsquared:.4f}")
m2.metric("Adj. R²", f"{result.rsquared_adj:.4f}")
m3.metric("F-statistic", f"{result.fvalue:,.1f}")
m4.metric("Observations", f"{int(result.nobs):,}")

st.markdown("**Full statsmodels summary**")
st.code(result.summary().as_text(), language="text")

st.markdown("**Coefficients**")
coef_table = pd.DataFrame(
    {
        "coef": result.params,
        "std err": result.bse,
        "t": result.tvalues,
        "P>|t|": result.pvalues,
        "[0.025": result.conf_int()[0],
        "0.975]": result.conf_int()[1],
    }
)
st.dataframe(coef_table.style.format("{:.6g}"), use_container_width=True)

# Multicollinearity note (mirrors the warning in the notebook output)
if result.condition_number > 1e6:
    st.warning(
        f"Condition number is large ({result.condition_number:.2e}). "
        "This may indicate multicollinearity or numerical problems "
        "(often caused by variables on very different scales, e.g. Volume vs. price)."
    )

# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
st.subheader("Residual diagnostics")
fitted = result.fittedvalues
resid = result.resid

d1, d2, d3 = st.columns(3)

with d1:
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.scatter(fitted, resid, s=8, alpha=0.5)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    st.pyplot(fig)
    plt.close(fig)

with d2:
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sm.qqplot(resid, line="s", ax=ax, markersize=3)
    ax.set_title("Normal Q-Q")
    st.pyplot(fig)
    plt.close(fig)

with d3:
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.hist(resid, bins=40, edgecolor="white")
    ax.set_title("Residual distribution")
    ax.set_xlabel("Residual")
    st.pyplot(fig)
    plt.close(fig)

st.markdown("**Actual vs Predicted**")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y.index, y.values, label="Actual", lw=1.2)
ax.plot(fitted.index, fitted.values, label="Predicted", lw=1.2, alpha=0.8)
ax.set_ylabel(target)
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Educational use only. Not financial advice. Regressing a price on same-day "
    "prices (e.g. Close on Open) will naturally give a very high R²."
)
