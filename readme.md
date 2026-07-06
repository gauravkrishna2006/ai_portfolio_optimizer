# 🤖 AI-Powered Portfolio Optimizer

An interactive **Machine Learning-powered portfolio optimization** web application built using **Streamlit**, **Scikit-learn**, and **PyPortfolioOpt**. The application predicts expected stock returns using a Random Forest model and generates an optimized portfolio allocation based on the user's risk appetite using Modern Portfolio Theory (MPT).

---

## 📌 Features

- 📈 Fetches historical stock market data from Yahoo Finance.
- 🤖 Predicts future stock returns using a Random Forest Regressor.
- 📊 Engineers technical indicators such as:
  - SMA 20 Ratio
  - SMA 50 Ratio
  - 20-Day Volatility
  - 10-Day Momentum
- 💼 Optimizes portfolio allocation using Efficient Frontier.
- ⚖️ Supports multiple risk profiles:
  - Low Risk (Minimum Volatility)
  - Medium Risk (Maximum Sharpe Ratio)
  - High Risk (Maximum Quadratic Utility)
- 📉 Displays expected return, portfolio volatility, and Sharpe ratio.
- 🥧 Interactive portfolio allocation visualization.
- 💰 Shows exact investment allocation for each selected stock.

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### Machine Learning
- Scikit-learn
- Random Forest Regressor

### Portfolio Optimization
- PyPortfolioOpt
- Modern Portfolio Theory (Efficient Frontier)

### Data Source
- Yahoo Finance API (yfinance)

### Data Processing
- Pandas
- NumPy

### Visualization
- Plotly

---

## 📂 Project Structure

```
AI-Portfolio-Optimizer/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Required dependencies
├── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/gauravkrishna2006/ai_portfolio_optimizer
```

Move into the project directory

```bash
cd ai_portfolio_optimizer
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 📊 How It Works

### Step 1 — Fetch Historical Data

The application downloads historical closing prices from Yahoo Finance for the selected stocks.

---

### Step 2 — Feature Engineering

For every stock, the following features are created:

- 20-Day Moving Average Ratio
- 50-Day Moving Average Ratio
- Daily Returns
- 20-Day Rolling Volatility
- 10-Day Momentum

These indicators are used as inputs for the machine learning model.

---

### Step 3 — Machine Learning Prediction

A **Random Forest Regressor** is trained individually for each stock to predict its future 20-day return.

The predicted return is annualized before portfolio optimization.

---

### Step 4 — Portfolio Optimization

The predicted returns are passed into the Efficient Frontier optimizer.

Depending on the selected risk profile:

| Risk Profile | Optimization Strategy |
|--------------|----------------------|
| Low | Minimum Volatility |
| Medium | Maximum Sharpe Ratio |
| High | Maximum Quadratic Utility |

Constraints applied:

- No Short Selling
- Maximum 40% allocation per stock
- L2 Regularization for diversification

---

### Step 5 — Dashboard

The application displays:

- Expected Annual Return
- Portfolio Volatility
- Sharpe Ratio
- Portfolio Allocation Chart
- Investment Breakdown

---

## 📸 Dashboard Preview

### Main Dashboard

- Portfolio Metrics
- Interactive Donut Chart
- Investment Allocation Table


---

## 📦 Dependencies

```
streamlit
yfinance
pandas
plotly
PyPortfolioOpt
scikit-learn
numpy
```

Install all using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Improvements

- Support Indian Stock Market (NSE/BSE)
- Portfolio Backtesting
- LSTM/XGBoost-based Return Prediction
- Risk Analysis using Value at Risk (VaR)
- Portfolio Comparison
- Sector-wise Diversification
- Real-time Market Updates
- Portfolio Performance Tracking

---

## 📈 Sample Workflow

```
Select Stocks
        │
        ▼
Download Historical Prices
        │
        ▼
Feature Engineering
        │
        ▼
Random Forest Prediction
        │
        ▼
Expected Returns
        │
        ▼
Efficient Frontier Optimization
        │
        ▼
Portfolio Allocation
        │
        ▼
Interactive Dashboard
```

---

## 👨‍💻 Author

**Gaurav Krishna**

B.Tech, Mathematics and Computing  
Indian Institute of Technology Jammu

LinkedIn: https://linkedin.com/in/gaurav-krishna-uma

---

## 📄 License

This project is intended for educational and research purposes.