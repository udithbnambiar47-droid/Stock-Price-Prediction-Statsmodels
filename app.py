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

st.set_page_config(page_title="Stock Price Prediction (OLS)", page_icon="📈", layout="wide")

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]

st.title("📈 Stock Price Prediction — OLS Regression")


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
    if isinstance(df.columns, pd.MultiIndex):  # flatten ('Close','GOOGL') -> 'Close'
        df.columns = df.columns.get_level_values(0)
    cols = [c for c in PRICE_COLS if c in df.columns]
    return df[cols].dropna()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features known at the close of day t, used to predict the close of day t+1."""
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"]
    out["Return"] = df["Close"].pct_change()
    out["Volume_M"] = df["Volume"] / 1e6  # scaled to keep the model well-conditioned
    out["Target"] = df["Close"].shift(-1)  # next trading day's close
    return out


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", value="GOOGL").strip().upper()
    years = st.slider("Years of history", min_value=1, max_value=15, value=5)
    st.divider()
    st.subheader("Custom OLS (explore)")
    target = st.selectbox("Dependent variable (y)", PRICE_COLS, index=3)
    default_x = [c for c in ["Open", "Volume"] if c != target]
    features = st.multiselect(
        "Independent variables (X)",
        [c for c in PRICE_COLS if c != target],
        default=default_x,
    )
    run = st.button("Run custom regression", type="primary", use_container_width=True)

if not ticker:
    st.info("Enter a ticker symbol in the sidebar.")
    st.stop()

data = load_data(ticker, years)
if data.empty or len(data) < 120:
    st.error(
        f"Not enough data returned for **{ticker}**. "
        "Check the ticker symbol or increase the years of history."
    )
    st.stop()

last_date = data.index[-1]
last_close = float(data["Close"].iloc[-1])
next_date = last_date + pd.offsets.BDay(1)  # next weekday (exchange holidays not modelled)

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
st.subheader(f"{ticker} — overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last close", f"${last_close:,.2f}")
c2.metric("As of", last_date.strftime("%d %b %Y"))
c3.metric("Rows", f"{len(data):,}")
c4.metric("History", f"{years} yr")

# --------------------------------------------------------------------------
# Next-day forecast
# --------------------------------------------------------------------------
st.header("🔮 Next trading day forecast")

feat_df = build_features(data)
FEATS = ["Close", "Return", "Volume_M"]

train_all = feat_df.dropna(subset=FEATS + ["Target"])
latest_row = feat_df.iloc[[-1]][FEATS]  # today's features -> tomorrow's close

# Out-of-sample check: train on first 80%, test on last 20%
split = int(len(train_all) * 0.8)
train, test = train_all.iloc[:split], train_all.iloc[split:]
bt_model = sm.OLS(train["Target"], sm.add_constant(train[FEATS])).fit()
bt_pred = bt_model.predict(sm.add_constant(test[FEATS], has_constant="add"))
mae_model = float(np.mean(np.abs(test["Target"] - bt_pred)))
mae_naive = float(np.mean(np.abs(test["Target"] - test["Close"])))  # "tomorrow = today"
mape_model = float(np.mean(np.abs((test["Target"] - bt_pred) / test["Target"])) * 100)

# Final model on all available history
fc_model = sm.OLS(train_all["Target"], sm.add_constant(train_all[FEATS])).fit()
x_last = sm.add_constant(latest_row, has_constant="add")
pred = fc_model.get_prediction(x_last).summary_frame(alpha=0.05)
fc_price = float(pred["mean"].iloc[0])
fc_low = float(pred["obs_ci_lower"].iloc[0])
fc_high = float(pred["obs_ci_upper"].iloc[0])
fc_change = fc_price - last_close
fc_pct = fc_change / last_close * 100

f1, f2, f3 = st.columns(3)
f1.metric(
    f"Predicted close — {next_date.strftime('%a, %d %b %Y')}",
    f"${fc_price:,.2f}",
    f"{fc_change:+.2f} ({fc_pct:+.2f}%)",
)
f2.metric("95% prediction range", f"${fc_low:,.2f} – ${fc_high:,.2f}")
f3.metric("Range width", f"±{(fc_high - fc_low) / 2 / last_close * 100:.1f}% of last close")

st.info(
    f"The model estimates **{ticker}** could close near **${fc_price:,.2f}** on the next "
    f"trading day, but there is a wide band of plausible outcomes "
    f"(**${fc_low:,.2f} – ${fc_high:,.2f}** at 95% confidence). "
    "Treat the number as a rough statistical estimate, not a promise.",
    icon="ℹ️",
)

# Forecast chart
recent = data["Close"].tail(60)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(recent.index, recent.values, label="Actual close (last 60 days)", lw=1.5)
ax.errorbar(
    [next_date],
    [fc_price],
    yerr=[[fc_price - fc_low], [fc_high - fc_price]],
    fmt="o",
    color="crimson",
    capsize=6,
    label="Forecast ± 95% range",
)
ax.plot([last_date, next_date], [last_close, fc_price], "--", color="crimson", alpha=0.6)
ax.set_ylabel("Price (USD)")
ax.set_title(f"{ticker}: next-day forecast with uncertainty range")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

with st.expander("How reliable is this forecast? (back-test on unseen data)", expanded=True):
    b1, b2, b3 = st.columns(3)
    b1.metric("Model avg. error (MAE)", f"${mae_model:,.2f}")
    b2.metric("'Tomorrow = today' error", f"${mae_naive:,.2f}")
    b3.metric("Model avg. error (%)", f"{mape_model:.2f}%")
    st.caption(
        f"Model trained on the first 80% of the history and tested on the most recent "
        f"{len(test)} trading days it had never seen. "
        "If the model's error is not clearly lower than the simple 'tomorrow = today' guess, "
        "it adds little real predictive value — which is common for stock prices."
    )
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(test.index, test["Target"].values, label="Actual", lw=1.2)
    ax.plot(test.index, bt_pred.values, label="Model prediction", lw=1.2, alpha=0.8)
    ax.set_title("Out-of-sample: actual vs predicted next-day close")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with st.expander("Forecast model details"):
    st.code(fc_model.summary().as_text(), language="text")
    st.caption("Model: Next-day Close = β0 + β1·Close + β2·Today's return + β3·Volume (millions) + ε")

# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------
st.header("🌊 Market volatility")
st.caption("Prices move up and down every day. This section shows how large those swings have been.")

ret = data["Close"].pct_change().dropna()
daily_sigma = float(ret.std())
ann_vol = daily_sigma * np.sqrt(252)
roll_vol = ret.rolling(20).std() * np.sqrt(252)
cum_max = data["Close"].cummax()
drawdown = data["Close"] / cum_max - 1
max_dd = float(drawdown.min())
big_moves = float((ret.abs() > 0.02).mean() * 100)

v1, v2, v3, v4 = st.columns(4)
v1.metric("Typical daily move (1σ)", f"±{daily_sigma * 100:.2f}%")
v2.metric("Annualised volatility", f"{ann_vol * 100:.1f}%")
v3.metric("Largest 1-day drop", f"{ret.min() * 100:.1f}%")
v4.metric("Largest 1-day gain", f"{ret.max() * 100:+.1f}%")

v5, v6, v7, v8 = st.columns(4)
v5.metric("Max drawdown", f"{max_dd * 100:.1f}%")
v6.metric("Days with move > ±2%", f"{big_moves:.0f}%")
v7.metric("Latest 20-day volatility", f"{roll_vol.dropna().iloc[-1] * 100:.1f}%")
v8.metric(
    "1σ range for next day",
    f"${last_close * (1 - daily_sigma):,.2f} – ${last_close * (1 + daily_sigma):,.2f}",
)

tab1, tab2, tab3 = st.tabs(["Price & rolling volatility", "Daily returns", "Drawdown"])

with tab1:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 1])
    a1.plot(data.index, data["Close"], lw=1.2)
    a1.set_ylabel("Close (USD)")
    a1.set_title(f"{ticker} price")
    a1.grid(alpha=0.3)
    a2.plot(roll_vol.index, roll_vol * 100, color="darkorange", lw=1.2)
    a2.axhline(ann_vol * 100, color="gray", ls="--", lw=1, label=f"Average: {ann_vol * 100:.1f}%")
    a2.set_ylabel("20-day vol (%)")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Higher orange line = more turbulent market and wider uncertainty on any forecast.")

with tab2:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(ret * 100, bins=60, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(-1.96 * daily_sigma * 100, color="crimson", ls="--", lw=1, label="±1.96σ")
    ax.axvline(1.96 * daily_sigma * 100, color="crimson", ls="--", lw=1)
    ax.set_xlabel("Daily return (%)")
    ax.set_ylabel("Number of days")
    ax.set_title("Distribution of daily returns")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("The long tails show that occasional very large moves do happen.")

with tab3:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.fill_between(drawdown.index, drawdown * 100, 0, color="crimson", alpha=0.4)
    ax.set_ylabel("Drop from previous peak (%)")
    ax.set_title("Drawdown from all-time high in this window")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"Worst peak-to-trough fall in this period: {max_dd * 100:.1f}%.")

# --------------------------------------------------------------------------
# Explore: correlation, pair plot, custom OLS
# --------------------------------------------------------------------------
st.header("🔬 Explore the data")

with st.expander("Preview data"):
    st.dataframe(data.tail(10), use_container_width=True)

st.subheader("Correlation matrix")
corr = data.corr()
col_a, col_b = st.columns(2)
with col_a:
    st.dataframe(corr.style.format("{:.4f}"), use_container_width=True)
with col_b:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Pair plot with regression lines")
pair_cols = [target] + features
if len(pair_cols) >= 2:
    if st.checkbox("Show pair plot (can take a few seconds)", value=False):
        g = sns.pairplot(
            data[pair_cols],
            kind="reg",
            plot_kws={"line_kws": {"color": "black"}, "scatter_kws": {"s": 8, "alpha": 0.5}},
        )
        st.pyplot(g.figure)
        plt.close(g.figure)
else:
    st.info("Select at least one independent variable to see the pair plot.")

st.subheader("Custom OLS regression")
if not features:
    st.warning("Select at least one independent variable in the sidebar.")
elif not run:
    st.info("Click **Run custom regression** in the sidebar to fit the model.")
else:
    y = data[target]
    X = sm.add_constant(data[features])
    result = sm.OLS(y, X).fit()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²", f"{result.rsquared:.4f}")
    m2.metric("Adj. R²", f"{result.rsquared_adj:.4f}")
    m3.metric("F-statistic", f"{result.fvalue:,.1f}")
    m4.metric("Observations", f"{int(result.nobs):,}")

    st.code(result.summary().as_text(), language="text")

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

    if result.condition_number > 1e6:
        st.warning(
            f"Condition number is large ({result.condition_number:.2e}), which may indicate "
            "multicollinearity or scaling problems (e.g. Volume vs. price)."
        )

    fitted, resid = result.fittedvalues, result.resid
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

    st.caption(
        "Note: regressing Close on same-day Open gives a very high R² simply because "
        "prices on the same day are close together. It cannot be used to forecast tomorrow."
    )

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.divider()
st.caption("Data source: Yahoo Finance via yfinance. Built with Streamlit and statsmodels.")
