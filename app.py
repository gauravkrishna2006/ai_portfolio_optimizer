import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- 1. CACHED BACKEND FUNCTIONS ---
# We use @st.cache_data so the app doesn't re-download data on every button click
@st.cache_data
def fetch_stock_data(tickers, start_date="2020-01-01", end_date="2026-01-01"):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data.dropna()

def engineer_features_and_predict(price_series):
    df = pd.DataFrame(price_series)
    df.columns = ['Close']
    
    df['SMA_20_Ratio'] = df['Close'] / df['Close'].rolling(window=20).mean()
    df['SMA_50_Ratio'] = df['Close'] / df['Close'].rolling(window=50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Vol_20'] = df['Daily_Return'].rolling(window=20).std()
    df['Momentum_10'] = df['Close'].pct_change(periods=10)
    df['Future_Return'] = df['Close'].pct_change(periods=20).shift(-20)
    
    latest_features = df.iloc[-1:][['SMA_20_Ratio', 'SMA_50_Ratio', 'Vol_20', 'Momentum_10']]
    df_clean = df.dropna()
    
    if df_clean.empty: return 0.0 
        
    X = df_clean[['SMA_20_Ratio', 'SMA_50_Ratio', 'Vol_20', 'Momentum_10']]
    y = df_clean['Future_Return']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    predicted_return = model.predict(latest_features)[0]
    return ((1 + predicted_return) ** (252 / 20)) - 1

@st.cache_data
def get_ml_expected_returns(historical_prices):
    ml_predictions = {}
    for ticker in historical_prices.columns:
        ml_predictions[ticker] = engineer_features_and_predict(historical_prices[ticker])
    return pd.Series(ml_predictions)

def optimize_portfolio(cleaned_data, mu, risk_appetite="Medium"):
    S = risk_models.sample_cov(cleaned_data)
    
    # Initialize with bounds: Minimum 0% (no shorting), Maximum 40% per asset
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, 0.40))
    ef.add_objective(objective_functions.L2_reg, gamma=0.5)
    
    # --- UPDATED RISK ROUTING ---
    if risk_appetite == "Low":
        # Strategy: Absolute lowest volatility possible (Capital Preservation)
        raw_weights = ef.min_volatility()
        
    elif risk_appetite == "Medium":
        # Strategy: Best risk-adjusted return (Max Sharpe Ratio)
        raw_weights = ef.max_sharpe(risk_free_rate=0.07)
        
    elif risk_appetite == "High":
        # Strategy: Aggressively chase returns (Low risk aversion penalty)
        raw_weights = ef.max_quadratic_utility(risk_aversion=0.1)
        
    cleaned_weights = ef.clean_weights()
    expected_return, volatility, sharpe_ratio = ef.portfolio_performance(verbose=False, risk_free_rate=0.07)
    
    return cleaned_weights, expected_return, volatility, sharpe_ratio

# --- 2. STREAMLIT FRONTEND UI ---
st.set_page_config(page_title="AI Portfolio Optimizer", layout="wide")
st.title("🤖 AI-Powered Portfolio Optimizer")
st.markdown("Optimize your investments using Machine Learning and Modern Portfolio Theory.")

# Sidebar Inputs
st.sidebar.header("User Preferences")
investment_amount = st.sidebar.number_input("Investment Amount ($)", min_value=100, value=10000, step=100)
risk_profile = st.sidebar.selectbox("Risk Appetite", ["Low", "Medium", "High"])
default_tickers = "AAPL, MSFT, GOOGL, AMZN, JPM, NVDA"
ticker_input = st.sidebar.text_input("Enter Tickers (comma-separated)", default_tickers)

if st.sidebar.button("Optimize Portfolio"):
    tickers = [t.strip().upper() for t in ticker_input.split(',')]
    
    with st.spinner('Fetching market data and training ML models...'):
        # 1. Fetch Data
        historical_prices = fetch_stock_data(tickers)
        
        # 2. Get ML Predictions
        ml_mu = get_ml_expected_returns(historical_prices)
        
        # 3. Optimize
        weights, exp_ret, vol, sharpe = optimize_portfolio(historical_prices, ml_mu, risk_appetite=risk_profile)
        
        # Filter out zero-weight assets
        active_weights = {k: v for k, v in weights.items() if v > 0}
        
    # --- 3. DASHBOARD VISUALIZATION ---
    st.success("Optimization Complete!")
    
    # Top Level Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Annual Return", f"{exp_ret * 100:.2f}%")
    col2.metric("Portfolio Risk (Volatility)", f"{vol * 100:.2f}%")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    # Visualizations and Exact Allocations
    st.subheader("Optimal Asset Allocation")
    col_chart, col_data = st.columns([2, 1])
    
    with col_chart:
        # Plotly Donut Chart
        fig = px.pie(
            values=list(active_weights.values()), 
            names=list(active_weights.keys()), 
            hole=0.4,
            title="Recommended Portfolio Weights"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
    with col_data:
        st.write("### Investment Breakdown")
        breakdown_data = []
        for ticker, weight in active_weights.items():
            allocated_money = investment_amount * weight
            breakdown_data.append({"Ticker": ticker, "Weight (%)": f"{weight*100:.2f}%", "Amount ($)": f"${allocated_money:.2f}"})
        
        st.dataframe(pd.DataFrame(breakdown_data), hide_index=True)