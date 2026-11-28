import warnings
from datetime import date, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]

# --------------------------------------------------------------------------
# Mobile-first styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding: 3.5rem 1rem 3rem 1rem; max-width: 1100px;}
    h1 {font-size: 1.6rem !important; line-height: 1.25 !important; padding: 0 0 .25rem 0 !important;}
    h2 {font-size: 1.25rem !important;}
    h3 {font-size: 1.05rem !important;}

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(128,128,128,.10);
        border-radius: 12px;
        padding: .6rem .8rem;
    }
    [data-testid="stMetricLabel"] p {font-size: .8rem; opacity: .8;}
    [data-testid="stMetricValue"] {font-size: 1.35rem;}
    [data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere;
    }

    /* Tabs: easy to tap, scroll sideways if needed */
    button[data-baseweb="tab"] {padding: .6rem .9rem; font-size: .95rem;}

    /* Phones: rows of metrics become a 2-column grid instead of one long column */
    @media (max-width: 640px) {
        .block-container {padding: 3.5rem .75rem 3rem .75rem;}
        div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
            flex-wrap: wrap !important;
            gap: .5rem !important;
        }
        div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div {
            flex: 1 1 calc(50% - .5rem) !important;
            min-width: calc(50% - .5rem) !important;
            width: auto !important;
        }
        [data-testid="stMetricValue"] {font-size: 1.2rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Inside the installed app (?embed=true) Streamlit hides its header, so drop the extra top gap
if str(st.query_params.get("embed", "")).lower() == "true":
    st.markdown(
        "<style>.block-container{padding-top:1rem !important}</style>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Currency handling
# --------------------------------------------------------------------------
# currency code -> (prefix, suffix)
CURRENCY_FORMATS = {
    "INR": ("₹", ""),
    "USD": ("$", ""),
    "EUR": ("€", ""),
    "GBP": ("£", ""),
    "GBp": ("", "p"),  # London quotes in pence
    "JPY": ("¥", ""),
    "CNY": ("¥", ""),
    "HKD": ("HK$", ""),
    "CAD": ("C$", ""),
    "AUD": ("A$", ""),
    "SGD": ("S$", ""),
    "KRW": ("₩", ""),
    "BRL": ("R$", ""),
    "CHF": ("CHF ", ""),
    "ZAR": ("R ", ""),
}
# Fallback when Yahoo does not return a currency: guess from the exchange suffix
SUFFIX_CURRENCY = {
    ".NS": "INR", ".BO": "INR", ".L": "GBp", ".T": "JPY", ".HK": "HKD",
    ".TO": "CAD", ".AX": "AUD", ".SI": "SGD", ".DE": "EUR", ".PA": "EUR",
    ".SS": "CNY", ".SZ": "CNY", ".KS": "KRW", ".SA": "BRL",
}


@st.cache_data(ttl=86400, show_spinner=False)
def get_currency(ticker: str) -> str:
    """Currency the stock is quoted in (e.g. INR for TCS.NS, USD for GOOGL)."""
    try:
        code = yf.Ticker(ticker).fast_info["currency"]
        if code:
            return str(code)
    except Exception:
        pass
    try:
        code = yf.Ticker(ticker).info.get("currency")
        if code:
            return str(code)
    except Exception:
        pass
    upper = ticker.upper()
    for suffix, code in SUFFIX_CURRENCY.items():
        if upper.endswith(suffix):
            return code
    return "USD"


def _indian_group(x: float) -> str:
    """12345678.9 -> '1,23,45,678.90' (lakh / crore grouping)."""
    whole, frac = f"{x:.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"{whole}.{frac}"


def money(x: float, cur: str) -> str:
    prefix, suffix = CURRENCY_FORMATS.get(cur, (f"{cur} ", ""))
    body = _indian_group(abs(x)) if cur == "INR" else f"{abs(x):,.2f}"
    return f"{'-' if x < 0 else ''}{prefix}{body}{suffix}"


def signed_money(x: float, cur: str) -> str:
    return f"{'+' if x >= 0 else '-'}{money(abs(x), cur)}"


def mmoney(x: float, cur: str) -> str:
    """money() made safe for st.markdown / st.info, where '$' starts a math block."""
    return money(x, cur).replace("$", r"\$")


# --------------------------------------------------------------------------
# Chart styling (follows the app's light / dark theme, phone-sized figures)
# --------------------------------------------------------------------------
def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


DARK = is_dark()
BG, FG, GRID = ("#0e1117", "#e6e6e6", "#3a3f4b") if DARK else ("#ffffff", "#262730", "#d9d9d9")
BLUE, RED, ORANGE = ("#4da3ff", "#ff6b6b", "#ffb454") if DARK else ("#1f77b4", "#d62728", "#e08a00")

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": FG,
        "axes.titlecolor": FG,
        "text.color": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "grid.color": GRID,
        "legend.facecolor": BG,
        "legend.edgecolor": GRID,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.autolayout": True,
    }
)
FIG_W = 6.4  # inches: scales down cleanly to a phone screen


def fix_dates(ax):
    loc = mdates.AutoDateLocator(minticks=3, maxticks=5)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
# Data
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
        raise ValueError("no data")  # raising keeps failed downloads out of the cache
    if isinstance(df.columns, pd.MultiIndex):  # flatten ('Close','TCS.NS') -> 'Close'
        df.columns = df.columns.get_level_values(0)
    cols = [c for c in PRICE_COLS if c in df.columns]
    return df[cols].dropna()


def try_load(ticker: str, years: int) -> pd.DataFrame:
    try:
        return load_data(ticker, years)
    except Exception:
        return pd.DataFrame()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features known at the close of day t, used to predict the close of day t+1."""
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"]
    out["Return"] = df["Close"].pct_change()
    out["Volume_M"] = df["Volume"] / 1e6  # scaled to keep the model well-conditioned
    out["Target"] = df["Close"].shift(-1)  # next trading day's close
    return out


# --------------------------------------------------------------------------
# Custom OLS helpers — no look-ahead: every predictor is strictly older than the target
# --------------------------------------------------------------------------
TARGETS = {"return": "Next-day return (%)", "close": "Next-day Close"}
MODEL_FEATURES = ["Return", "Volume change", "Close", "High", "Low", "Open", "Volume"]
FEATURE_LABELS = {
    "Return": "Return (%)",
    "Volume change": "Volume change (%)",
    "Close": "Close",
    "High": "High",
    "Low": "Low",
    "Open": "Open",
    "Volume": "Volume (millions)",
}
LAG_CHOICES = [1, 2, 3, 5]
TRAIN_FRACTION = 0.8


def build_model_frame(df: pd.DataFrame, features: list, n_lags: int, target_kind: str) -> pd.DataFrame:
    """Modelling table. Row t is a trading day we want to forecast; the model only sees data up to
    the close of day t-1, because every predictor is shifted back by at least one day.

    Columns: 'target' (what happens on day t), 'naive' (the baseline forecast for day t) and one
    '<feature>_lag<k>' column per feature and lag k = 1..n_lags. No same-day predictor exists.
    """
    if n_lags < 1:
        raise ValueError("Predictors must be lagged by at least one day.")
    close = df["Close"]
    vol = df["Volume"].astype(float)
    base = {
        "Close": close,
        "High": df["High"],
        "Low": df["Low"],
        "Open": df["Open"],
        "Volume": vol / 1e6,  # millions: keeps the design matrix well scaled
        "Return": close.pct_change() * 100,
        "Volume change": vol.pct_change().replace([np.inf, -np.inf], np.nan) * 100,
    }
    lagged = {f"{name}_lag{k}": base[name].shift(k) for name in features for k in range(1, n_lags + 1)}
    if target_kind == "close":
        target = close  # next day's Close, relative to the last day the model has seen
        naive = close.shift(1)  # "tomorrow = today"
    else:
        target = close.pct_change() * 100  # next day's return (%)
        naive = pd.Series(0.0, index=df.index)  # "no change"
    frame = pd.DataFrame({"target": target, "naive": naive, **lagged}, index=df.index)
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    err = y_true - y_pred
    sse = float((err**2).sum())
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "r2": 1 - sse / sst if sst > 0 else float("nan"),
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(err.abs().mean()),
    }


def adf_pvalue(series: pd.Series):
    """Augmented Dickey-Fuller p-value (small = stationary). None if the test cannot run."""
    try:
        return float(adfuller(series.dropna(), autolag="AIC")[1])
    except Exception:
        return None


def standardized_condition_number(X: pd.DataFrame) -> float:
    """Scale-free multicollinearity measure: condition number of the z-scored predictors."""
    std = X.std(ddof=0).replace(0, np.nan)
    Z = ((X - X.mean()) / std).dropna(axis=1, how="any")  # constant columns drop out
    return float(np.linalg.cond(Z.to_numpy())) if Z.shape[1] >= 2 else 1.0


# --------------------------------------------------------------------------
# Header + controls (in the main page: the sidebar is easy to miss on phones)
# --------------------------------------------------------------------------
st.title("📈 Stock Price Prediction")
st.caption("OLS regression on Yahoo Finance data")

ctl1, ctl2 = st.columns([3, 2])
with ctl1:
    ticker = st.text_input(
        "Ticker symbol",
        value="GOOGL",
        help="US stocks: GOOGL, AAPL. Indian stocks: add .NS (NSE) or .BO (BSE), e.g. TCS.NS, RELIANCE.NS",
    ).strip().upper()
with ctl2:
    years = st.selectbox("History", [1, 2, 3, 5, 10, 15], index=3, format_func=lambda y: f"{y} year{'s' if y > 1 else ''}")
st.caption("Indian stocks: add **.NS** (NSE) or **.BO** (BSE) — e.g. `TCS.NS`, `RELIANCE.NS`.")

if not ticker:
    st.info("Enter a ticker symbol above.")
    st.stop()

data = try_load(ticker, years)

# Friendly fallback: "TCS" -> try "TCS.NS" before giving up
if data.empty and "." not in ticker and "-" not in ticker and "=" not in ticker:
    alt = f"{ticker}.NS"
    alt_data = try_load(alt, years)
    if not alt_data.empty:
        st.info(f"No listing found for **{ticker}**, showing **{alt}** (NSE, India).")
        ticker, data = alt, alt_data

if data.empty or len(data) < 120:
    st.error(
        f"Not enough data for **{ticker}**. Check the symbol "
        "(Indian stocks need `.NS` or `.BO`) or choose a longer history."
    )
    st.stop()

CUR = get_currency(ticker)

last_date = data.index[-1]
last_close = float(data["Close"].iloc[-1])
prev_close = float(data["Close"].iloc[-2])
day_chg = last_close - prev_close
next_date = last_date + pd.offsets.BDay(1)  # next weekday (exchange holidays not modelled)

st.subheader(f"{ticker} — overview")
o1, o2, o3, o4 = st.columns(4)
o1.metric("Last close", money(last_close, CUR), f"{signed_money(day_chg, CUR)} ({day_chg / prev_close * 100:+.2f}%)")
o2.metric("As of", last_date.strftime("%d %b %Y"))
o3.metric("Trading days", f"{len(data):,}")
o4.metric("Currency", CUR)

tab_fc, tab_vol, tab_ex = st.tabs(["🔮 Forecast", "🌊 Volatility", "🔬 Explore"])

# ==========================================================================
# TAB 1 — Next-day forecast
# ==========================================================================
with tab_fc:
    st.subheader("Next trading day")

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
    half_width_pct = (fc_high - fc_low) / 2 / last_close * 100

    st.metric(
        f"Predicted close · {next_date.strftime('%a, %d %b %Y')}",
        money(fc_price, CUR),
        f"{signed_money(fc_change, CUR)} ({fc_pct:+.2f}%)",
    )
    lo, hi = st.columns(2)
    lo.metric("Low end (95%)", money(fc_low, CUR))
    hi.metric("High end (95%)", money(fc_high, CUR))

    st.info(
        f"**{ticker}** could close near **{mmoney(fc_price, CUR)}** on the next trading day, "
        f"but outcomes within about ±{half_width_pct:.1f}% of today's close are plausible "
        f"({mmoney(fc_low, CUR)} to {mmoney(fc_high, CUR)}). "
        "Treat this as a rough statistical estimate, not a promise.",
        icon="ℹ️",
    )

    recent = data["Close"].tail(60)
    fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
    ax.plot(recent.index, recent.values, color=BLUE, lw=1.6, label="Close (last 60 days)")
    ax.errorbar(
        [next_date], [fc_price],
        yerr=[[fc_price - fc_low], [fc_high - fc_price]],
        fmt="o", color=RED, capsize=5, label="Forecast ± 95%",
    )
    ax.plot([last_date, next_date], [last_close, fc_price], "--", color=RED, alpha=0.6)
    ax.set_ylabel(f"Price ({CUR})")
    ax.set_title(f"{ticker}: next-day forecast")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.3)
    fix_dates(ax)
    show(fig)

    with st.expander("How reliable is this? (back-test)", expanded=True):
        b1, b2, b3 = st.columns(3)
        b1.metric("Model avg. error", money(mae_model, CUR))
        b2.metric("'Tomorrow = today' error", money(mae_naive, CUR))
        b3.metric("Model error %", f"{mape_model:.2f}%")
        st.caption(
            f"Trained on the first 80% of history, tested on the latest {len(test)} trading days it "
            "had never seen. If the model's error is not clearly lower than the simple "
            "'tomorrow = today' guess, it adds little real predictive value — common for stock prices."
        )
        fig, ax = plt.subplots(figsize=(FIG_W, 3.2))
        ax.plot(test.index, test["Target"].values, color=BLUE, lw=1.2, label="Actual")
        ax.plot(test.index, bt_pred.values, color=ORANGE, lw=1.2, alpha=0.9, label="Model")
        ax.set_title("Out-of-sample: actual vs predicted")
        ax.legend(loc="best", frameon=False)
        ax.grid(alpha=0.3)
        fix_dates(ax)
        show(fig)

    with st.expander("Model details"):
        st.code(fc_model.summary().as_text(), language="text")
        st.caption("Next-day Close = β0 + β1·Close + β2·Today's return + β3·Volume (millions) + ε")

# ==========================================================================
# TAB 2 — Volatility
# ==========================================================================
with tab_vol:
    st.subheader("Market volatility")
    st.caption("Prices move up and down every day. These numbers show how large the swings have been.")

    ret = data["Close"].pct_change().dropna()
    daily_sigma = float(ret.std())
    ann_vol = daily_sigma * np.sqrt(252)
    roll_vol = ret.rolling(20).std() * np.sqrt(252)
    drawdown = data["Close"] / data["Close"].cummax() - 1
    max_dd = float(drawdown.min())
    big_moves = float((ret.abs() > 0.02).mean() * 100)

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Typical daily move", f"±{daily_sigma * 100:.2f}%")
    v2.metric("Annual volatility", f"{ann_vol * 100:.1f}%")
    v3.metric("Worst day", f"{ret.min() * 100:.1f}%")
    v4.metric("Best day", f"{ret.max() * 100:+.1f}%")

    v5, v6, v7, v8 = st.columns(4)
    v5.metric("Max drawdown", f"{max_dd * 100:.1f}%")
    v6.metric("Days moving > ±2%", f"{big_moves:.0f}%")
    v7.metric("Latest 20-day vol", f"{roll_vol.dropna().iloc[-1] * 100:.1f}%")
    v8.metric("Typical range tomorrow", f"±{money(last_close * daily_sigma, CUR)}")

    sub1, sub2, sub3 = st.tabs(["Volatility", "Daily returns", "Drawdown"])

    with sub1:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(FIG_W, 5.0), sharex=True, height_ratios=[2, 1])
        a1.plot(data.index, data["Close"], color=BLUE, lw=1.2)
        a1.set_ylabel(f"Close ({CUR})")
        a1.set_title(f"{ticker} price")
        a1.grid(alpha=0.3)
        a2.plot(roll_vol.index, roll_vol * 100, color=ORANGE, lw=1.2)
        a2.axhline(ann_vol * 100, color=FG, ls="--", lw=0.8, alpha=0.6, label=f"Average {ann_vol * 100:.1f}%")
        a2.set_ylabel("20-day vol (%)")
        a2.legend(loc="upper left", frameon=False)
        a2.grid(alpha=0.3)
        fix_dates(a2)
        show(fig)
        st.caption("A higher orange line means a more turbulent market and wider uncertainty on any forecast.")

    with sub2:
        fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
        ax.hist(ret * 100, bins=50, color=BLUE, edgecolor=BG)
        ax.axvline(0, color=FG, lw=0.8)
        ax.axvline(-1.96 * daily_sigma * 100, color=RED, ls="--", lw=1, label="±1.96σ")
        ax.axvline(1.96 * daily_sigma * 100, color=RED, ls="--", lw=1)
        ax.set_xlabel("Daily return (%)")
        ax.set_ylabel("Number of days")
        ax.set_title("Distribution of daily returns")
        ax.legend(frameon=False)
        show(fig)
        st.caption("The long tails show that occasional very large moves do happen.")

    with sub3:
        fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
        ax.fill_between(drawdown.index, drawdown * 100, 0, color=RED, alpha=0.45)
        ax.set_ylabel("Drop from peak (%)")
        ax.set_title("Drawdown")
        ax.grid(alpha=0.3)
        fix_dates(ax)
        show(fig)
        st.caption(f"Worst peak-to-trough fall in this period: {max_dd * 100:.1f}%.")

# ==========================================================================
# TAB 3 — Explore: data, correlation, pair plot, custom OLS
# ==========================================================================
with tab_ex:
    st.subheader("Recent data")
    recent_tbl = data.tail(10).round(2)
    recent_tbl.index = recent_tbl.index.strftime("%d %b")  # short dates fit a phone screen
    st.dataframe(recent_tbl)

    # ---- CSV downloads: daily prices for a chosen year, or one row per year ----
    st.markdown("**Download data (CSV)**")
    dl_kind = st.radio(
        "What to download",
        ["Daily prices", "Yearly summary"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if dl_kind == "Daily prices":
        years_available = sorted({d.year for d in data.index}, reverse=True)
        year_pick = st.selectbox(
            "Year",
            ["All years"] + years_available,
            format_func=str,
            help="The years listed depend on the History you chose at the top of the page.",
        )
        sub = data if year_pick == "All years" else data[data.index.year == year_pick]
        daily_csv = sub.round(4).copy()
        daily_csv.insert(0, "Date", daily_csv.index.strftime("%Y-%m-%d"))
        daily_csv["Currency"] = CUR
        label_year = "all-years" if year_pick == "All years" else str(year_pick)
        st.caption(
            f"{len(sub):,} trading days · {sub.index.min():%d %b %Y} to {sub.index.max():%d %b %Y}"
            + (" · year in progress" if year_pick == last_date.year else "")
        )
        st.download_button(
            f"⬇️ Download {ticker} {label_year} (CSV)",
            data=daily_csv.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_{label_year}_daily.csv",
            mime="text/csv",
            type="primary",
        )
    else:
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        agg = {c: f for c, f in agg.items() if c in data.columns}
        yearly = data.groupby(data.index.year).agg(agg)
        yearly["Trading days"] = data.groupby(data.index.year).size()
        yearly["Change %"] = yearly["Close"].pct_change() * 100  # vs previous year's last close
        yearly.index.name = "Year"
        st.dataframe(yearly.round(2))
        st.caption(
            "One row per year: first open, highest high, lowest low, last close and total volume. "
            "Change % compares each year's last close with the previous year's, so the first year is blank "
            f"and {last_date.year} is year-to-date."
        )
        yearly_csv = yearly.round(4).reset_index()
        yearly_csv["Currency"] = CUR
        st.download_button(
            f"⬇️ Download {ticker} yearly summary (CSV)",
            data=yearly_csv.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_yearly_summary.csv",
            mime="text/csv",
            type="primary",
        )

    with st.expander("Summary statistics"):
        st.dataframe(data.describe().round(2))

    st.subheader("Correlation matrix")
    corr = data.corr()
    st.dataframe(corr.style.format("{:.4f}"))
    fig, ax = plt.subplots(figsize=(FIG_W * 0.8, 3.6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
                cbar=False, annot_kws={"size": 8})
    show(fig)

    st.subheader("Custom OLS regression")
    st.info(
        "**Why a price-level regression shows R² ≈ 0.999, and why this one doesn't**\n\n"
        "- **Leakage:** a day's Open, High and Low are almost the same number as its Close, so using them "
        "to \"predict\" Close just copies the answer.\n"
        "- **Non-stationarity:** prices drift upward or downward, and any two drifting series look related "
        "(*spurious regression*). That inflates R² and makes the coefficients meaningless.\n"
        "- **What this model does instead:** it only uses data up to the *previous* close to predict the next "
        "day, is fitted on the first 80% of dates and is scored on the last 20%, which it never saw.\n"
        "- **A low R² on returns is normal.** Daily returns are close to random, so an out-of-sample R² near "
        "zero (even slightly negative) is what an honest model looks like. Judge it against the baseline "
        "below, not against 0.999."
    )

    target_kind = st.selectbox("Dependent variable (y)", list(TARGETS), index=0, format_func=TARGETS.get)
    default_x = ["Return", "Volume change"] if target_kind == "return" else ["Close", "Return", "Volume change"]
    features = st.multiselect(
        "Independent variables (X), lagged",
        MODEL_FEATURES,
        default=default_x,
        format_func=FEATURE_LABELS.get,
        key=f"ols_features_{target_kind}",
        help="Every predictor is shifted back by at least one day, so the model never sees the day it predicts.",
    )
    n_lags = st.selectbox(
        "Days of history per predictor",
        LAG_CHOICES,
        format_func=lambda n: "Yesterday only" if n == 1 else f"Last {n} days",
    )

    ready = False
    if not features:
        st.warning("Select at least one independent variable.")
    else:
        frame = build_model_frame(data, features, n_lags, target_kind)
        feature_cols = [c for c in frame.columns if c not in ("target", "naive")]
        split = int(len(frame) * TRAIN_FRACTION)  # chronological: first 80% train, last 20% test
        train, test = frame.iloc[:split], frame.iloc[split:]
        ready = len(train) >= len(feature_cols) + 30 and len(test) >= 20
        if not ready:
            st.warning(
                "Not enough rows after lagging. Choose fewer predictors or lags, or a longer History at the top."
            )

    if ready:
        if st.checkbox("Show pair plot (slower)", value=False):
            pair_cols = [c for c in feature_cols if c.endswith("_lag1")][:4]
            pair_df = frame[["target"] + pair_cols].rename(columns={"target": TARGETS[target_kind]})
            g = sns.pairplot(
                pair_df,
                kind="reg",
                height=2.0,
                plot_kws={"line_kws": {"color": RED}, "scatter_kws": {"s": 6, "alpha": 0.5}},
            )
            show(g.figure)
            st.caption("The target against each predictor's 1-day lag (first four predictors).")

        # ---- fit on the training period only, score on the unseen test period ----
        y_name = "next_day_return_pct" if target_kind == "return" else "next_day_close"
        result = sm.OLS(train["target"].rename(y_name), sm.add_constant(train[feature_cols])).fit()
        pred_test = result.predict(sm.add_constant(test[feature_cols], has_constant="add"))
        resid_test = test["target"] - pred_test
        m_test = regression_metrics(test["target"], pred_test)
        m_base = regression_metrics(test["target"], test["naive"])

        if target_kind == "close":
            err = lambda v: money(v, CUR)  # noqa: E731
            base_label = "Baseline: tomorrow = today"
            y_label = f"Close ({CUR})"
        else:
            err = lambda v: f"{v:.3f}%"  # noqa: E731
            base_label = "Baseline: 0% return"
            y_label = "Daily return (%)"

        st.caption(
            f"Train: {train.index.min():%d %b %Y} to {train.index.max():%d %b %Y} ({len(train):,} days) · "
            f"Test: {test.index.min():%d %b %Y} to {test.index.max():%d %b %Y} ({len(test):,} days). "
            "Chronological split, no shuffling; the model is fitted on the training days only."
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Train R²", f"{result.rsquared:.4f}")
        r2.metric("Test R²", f"{m_test['r2']:.4f}")
        r3.metric("Test RMSE", err(m_test["rmse"]))
        r4.metric("Test MAE", err(m_test["mae"]))

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Adj. R² (train)", f"{result.rsquared_adj:.4f}")
        q2.metric("F-statistic (train)", f"{result.fvalue:,.1f}")
        q3.metric("Train days", f"{int(result.nobs):,}")
        q4.metric("Test days", f"{len(test):,}")

        # ---- baseline comparison ----
        st.markdown("**Model vs naive baseline (test)**")
        st.dataframe(
            pd.DataFrame(
                {
                    "RMSE": [err(m_test["rmse"]), err(m_base["rmse"])],
                    "MAE": [err(m_test["mae"]), err(m_base["mae"])],
                },
                index=["Model", base_label],
            )
        )
        gain_rmse = (m_base["rmse"] - m_test["rmse"]) / m_base["rmse"] * 100
        gain_mae = (m_base["mae"] - m_test["mae"]) / m_base["mae"] * 100
        if gain_rmse > 0 and gain_mae > 0:
            st.success(
                f"The model beats the baseline on the test period: RMSE {gain_rmse:.1f}% lower and "
                f"MAE {gain_mae:.1f}% lower."
            )
        elif gain_rmse <= 0 and gain_mae <= 0:
            st.warning(
                f"The model does **not** beat the baseline (RMSE {abs(gain_rmse):.1f}% higher, "
                f"MAE {abs(gain_mae):.1f}% higher). It adds nothing over simply assuming "
                f"{'tomorrow equals today' if target_kind == 'close' else 'a 0% return'}."
            )
        else:
            st.info(
                f"Mixed result: RMSE is {abs(gain_rmse):.1f}% {'lower' if gain_rmse > 0 else 'higher'} and MAE is "
                f"{abs(gain_mae):.1f}% {'lower' if gain_mae > 0 else 'higher'} than the baseline, "
                "so there is no clear advantage."
            )
        st.caption("One test window can flatter or punish a model, so treat small differences as noise.")

        # ---- diagnostics ----
        st.markdown("**Diagnostics**")
        dw = float(durbin_watson(result.resid))
        adf_p = adf_pvalue(train["target"])
        g1, g2 = st.columns(2)
        g1.metric("Durbin-Watson", f"{dw:.2f}", help="Residual autocorrelation. About 2 = none; below 1.5 = positive; above 2.5 = negative.")
        g2.metric(
            "ADF p-value (target)",
            "n/a" if adf_p is None else f"{adf_p:.3f}",
            help="Augmented Dickey-Fuller test on the training target. Below 0.05 = the series looks stationary.",
        )
        non_stationary = adf_p is not None and adf_p > 0.05
        autocorrelated = dw < 1.5 or dw > 2.5
        if non_stationary:
            st.warning(
                f"The target looks **non-stationary** (ADF p = {adf_p:.2f}, above 0.05). R² and coefficients on "
                "trending levels can be inflated or spurious. Use next-day returns instead."
            )
        if autocorrelated:
            st.warning(
                f"The residuals look **autocorrelated** (Durbin-Watson = {dw:.2f}; about 2 is healthy). "
                "Standard errors and p-values may be unreliable."
            )
        if not non_stationary and not autocorrelated:
            st.caption(
                f"Diagnostics look fine: the target looks stationary (ADF p = "
                f"{'n/a' if adf_p is None else f'{adf_p:.3f}'}) and the residuals show no strong autocorrelation "
                f"(Durbin-Watson = {dw:.2f})."
            )

        cond = standardized_condition_number(train[feature_cols])
        if cond > 30:
            st.warning(
                f"Possible multicollinearity: the condition number of the standardised predictors is {cond:,.0f} "
                "(above about 30 is a concern). A common cause is Close, High, Low and Open being almost the same "
                "number, which makes their lags interchangeable. Keep just one of them, or use returns."
            )

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
        st.dataframe(coef_table.style.format("{:.4g}"))
        st.caption("Coefficients come from the training period. Swipe the table sideways to see p-values and confidence intervals.")

        with st.expander("Full statsmodels summary"):
            st.code(result.summary().as_text(), language="text")

        # ---- residual plots ----
        fitted, resid = result.fittedvalues, result.resid
        d1, d2, d3, d4 = st.tabs(["Residuals", "Over time", "Q-Q plot", "Histogram"])
        with d1:
            fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
            ax.scatter(fitted, resid, s=6, alpha=0.5, color=BLUE)
            ax.axhline(0, color=FG, lw=0.8)
            ax.set_xlabel("Fitted values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs fitted (training)")
            show(fig)
        with d2:
            fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
            ax.plot(resid.index, resid.values, color=BLUE, lw=0.8, label="Train")
            ax.plot(resid_test.index, resid_test.values, color=ORANGE, lw=0.8, label="Test")
            ax.axhline(0, color=FG, lw=0.8)
            ax.axvline(test.index[0], color=RED, ls="--", lw=1)
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals over time")
            ax.legend(loc="best", frameon=False)
            ax.grid(alpha=0.3)
            fix_dates(ax)
            show(fig)
            st.caption("Runs of same-sign residuals mean autocorrelation. The dashed line is where the test period starts.")
        with d3:
            fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
            sm.qqplot(resid, line="s", ax=ax, markersize=3)
            ax.set_title("Normal Q-Q (training)")
            show(fig)
        with d4:
            fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
            ax.hist(resid, bins=40, color=BLUE, edgecolor=BG)
            ax.set_xlabel("Residual")
            ax.set_title("Residual distribution (training)")
            show(fig)

        # ---- actual vs predicted, on the unseen test period ----
        st.subheader("Actual vs predicted (test period)")
        a1, a2 = st.tabs(["Over time", "Scatter"])
        with a1:
            fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
            ax.plot(test.index, test["target"], color=BLUE, lw=1.0, label="Actual")
            ax.plot(pred_test.index, pred_test.values, color=ORANGE, lw=1.2, alpha=0.9, label="Model")
            ax.plot(test.index, test["naive"], color=FG, lw=0.9, ls="--", alpha=0.6, label=base_label.replace("Baseline: ", "Baseline "))
            ax.set_ylabel(y_label)
            ax.set_title("Actual vs predicted")
            ax.legend(loc="best", frameon=False)
            ax.grid(alpha=0.3)
            fix_dates(ax)
            show(fig)
        with a2:
            fig, ax = plt.subplots(figsize=(FIG_W, 3.8))
            ax.scatter(pred_test, test["target"], s=8, alpha=0.5, color=BLUE)
            lo = float(min(pred_test.min(), test["target"].min()))
            hi = float(max(pred_test.max(), test["target"].max()))
            ax.plot([lo, hi], [lo, hi], color=RED, lw=1, ls="--", label="Perfect prediction")
            ax.set_xlabel(f"Predicted ({y_label})")
            ax.set_ylabel(f"Actual ({y_label})")
            ax.set_title("Actual vs predicted (test)")
            ax.legend(loc="best", frameon=False)
            ax.grid(alpha=0.3)
            show(fig)
        st.caption("Train metrics describe the past. The test metrics and the baseline comparison are the honest score.")

st.divider()
st.caption("Data: Yahoo Finance via yfinance · Built with Streamlit and statsmodels.")
