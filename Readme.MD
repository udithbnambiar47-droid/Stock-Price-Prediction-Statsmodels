# 📈 Stock Price Prediction using Statsmodels (OLS Regression)

An interactive Streamlit app that downloads historical stock data from Yahoo Finance and fits an Ordinary Least Squares (OLS) regression model using `statsmodels`.

## 🔗 Live Demo

**[Open the app](https://stock-price-prediction-statsmodels-les75xtd9cpywfjfmaqui4.streamlit.app/)**

https://stock-price-prediction-statsmodels-les75xtd9cpywfjfmaqui4.streamlit.app/

## ✨ Features

- Download up to 15 years of price history for any ticker (default: `GOOGL`)
- Data preview and summary metrics
- Correlation matrix (table + heatmap)
- Pair plot with regression lines
- Choose your own dependent (y) and independent (X) variables
- Full OLS summary: R², Adjusted R², F-statistic, coefficients, p-values and confidence intervals
- Residual diagnostics: residuals vs fitted, Q-Q plot, residual histogram
- Actual vs predicted chart
- Multicollinearity warning when the condition number is large

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [statsmodels](https://www.statsmodels.org/)
- pandas, NumPy, SciPy
- matplotlib, seaborn

## 📁 Project Structure

```
.
├── app.py             # Streamlit application
├── requirements.txt   # Python dependencies
├── statsmodels.ipynb  # Original analysis notebook (optional)
└── README.md
```

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 📖 How to Use

1. Enter a ticker symbol (e.g. `GOOGL`, `AAPL`, `MSFT`) in the sidebar.
2. Select the number of years of history.
3. Choose the dependent variable and one or more independent variables.
4. Click **Run regression** to view the OLS results and diagnostics.

## 🧮 Model

The default model is:

```
Close = β0 + β1 · Open + β2 · Volume + ε
```

Because `Open` and `Close` are highly correlated, the R² is very high (~0.999). This reflects same-day price co-movement and is **not** evidence of real predictive power.

## ⚠️ Disclaimer

This project is for educational purposes only and is **not financial advice**. Do not make investment decisions based on this model.

## 📄 License

Released under the MIT License.
