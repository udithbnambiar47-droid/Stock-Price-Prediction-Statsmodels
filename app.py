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

    st.subheader("Correlation")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(FIG_W * 0.8, 3.6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
                cbar=False, annot_kws={"size": 8})
    show(fig)

    st.subheader("Custom OLS regression")
    target = st.selectbox("Dependent variable (y)", PRICE_COLS, index=3)
    default_x = [c for c in ["Open", "Volume"] if c != target]
    features = st.multiselect(
        "Independent variables (X)",
        [c for c in PRICE_COLS if c != target],
        default=default_x,
    )

    if len(features) >= 1 and st.checkbox("Show pair plot (slower)", value=False):
        g = sns.pairplot(
            data[[target] + features],
            kind="reg",
            height=2.0,
            plot_kws={"line_kws": {"color": RED}, "scatter_kws": {"s": 6, "alpha": 0.5}},
        )
        show(g.figure)

    if not features:
        st.warning("Select at least one independent variable.")
    else:
        result = sm.OLS(data[target], sm.add_constant(data[features])).fit()

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("R²", f"{result.rsquared:.4f}")
        r2.metric("Adj. R²", f"{result.rsquared_adj:.4f}")
        r3.metric("F-statistic", f"{result.fvalue:,.1f}")
        r4.metric("Observations", f"{int(result.nobs):,}")

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
        st.dataframe(coef_table.style.format("{:.6g}"))

        if result.condition_number > 1e6:
            st.warning(
                f"Condition number is large ({result.condition_number:.2e}), which may indicate "
                "multicollinearity or scaling problems (e.g. Volume vs. price)."
            )

        with st.expander("Full statsmodels summary"):
            st.code(result.summary().as_text(), language="text")

        fitted, resid = result.fittedvalues, result.resid
        d1, d2, d3 = st.tabs(["Residuals", "Q-Q plot", "Histogram"])
        with d1:
            fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
            ax.scatter(fitted, resid, s=6, alpha=0.5, color=BLUE)
            ax.axhline(0, color=FG, lw=0.8)
            ax.set_xlabel("Fitted values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs fitted")
            show(fig)
        with d2:
            fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
            sm.qqplot(resid, line="s", ax=ax, markersize=3)
            ax.set_title("Normal Q-Q")
            show(fig)
        with d3:
            fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
            ax.hist(resid, bins=40, color=BLUE, edgecolor=BG)
            ax.set_xlabel("Residual")
            ax.set_title("Residual distribution")
            show(fig)

        st.caption(
            "Note: regressing Close on same-day Open gives a very high R² simply because prices "
            "on the same day are close together. It cannot be used to forecast tomorrow."
        )

st.divider()
st.caption("Data: Yahoo Finance via yfinance · Built with Streamlit and statsmodels.")
