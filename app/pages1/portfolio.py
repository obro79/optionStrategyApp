import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Efficient Frontier Function
def calculate_efficient_frontier(returns, num_portfolios=5000, risk_free_rate=0.01):
    num_assets = returns.shape[1]
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    # Check if there are enough data points
    if returns.shape[0] < 2 or returns.shape[1] < 2:
        st.error("Either an invalid Ticker was entered or there was an error fetching the data")
        return None, None
    
    for i in range(num_portfolios):
        # Random weights for each asset
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Normalize weights
        weights_record.append(weights)
        
        # Portfolio return and risk
        portfolio_return = np.sum(weights * returns.mean()) * 252  # Annualized return

        # Check for divide-by-zero issues or NaN values in covariance calculation
        try:
            portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns.T) * 252, weights)))  # Annualized risk
        except Exception as e:
            st.error(f"Error in calculating portfolio standard deviation: {e}")
            continue
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
        
        # Store results
        results[0, i] = portfolio_stddev
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio
    
    return results, weights_record

# Portfolio volatility function
def portfolio_volatility(weights, mean_returns, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Function to fetch data from Yahoo Finance
def fetch_yahoo_data(tickers):
    data = yf.download(tickers, start="2020-01-01", end="2024-01-01")['Adj Close']
    return data

# Markowitz Efficient Frontier - Optimization Function
def portfolio_optimization(returns, target_return):
    num_assets = returns.shape[1]
    mean_returns = returns.mean() * 252  # Annualized mean returns
    cov_matrix = returns.cov() * 252     # Annualized covariance matrix

    # Objective function (minimize portfolio volatility)
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Constraints: sum of weights = 1 and target return = desired return
    constraints = ({
        'type': 'eq',
        'fun': lambda x: np.sum(x) - 1
    }, {
        'type': 'eq',
        'fun': lambda x: np.sum(x * mean_returns) - target_return
    })

    # Bounds: each weight is between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Initial guess (equal allocation to all assets)
    initial_guess = num_assets * [1. / num_assets,]

    # Minimize the portfolio volatility to find the optimal weights
    result = minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    return result


# Portfolio Management Page
def portfolio_management_page():

    
    st.title("Portfolio Management: Efficient Frontier")
    
    num_stocks = st.sidebar.number_input("Number of Stocks", min_value=2, max_value=10, value=2)


    # Predefined stocks to load initially
    default_stocks = ['AAPL', 'MSFT']

    tickers_list = []
    for i in range(num_stocks):
        stock_ticker = st.sidebar.text_input(f"Stock {i+1}", value=f"Stock {i+1}" if i >= 2 else ['AAPL', 'MSFT'][i])  # Default tickers for first two stocks
        tickers_list.append(stock_ticker)


    # Fetch the historical data for the stocks
    st.write(f"Optimizing for: {tickers_list}")
    data = fetch_yahoo_data(tickers_list)
    
    if data is not None:
        returns = data.pct_change().dropna()  # Calculate returns from price data

        # Plot Efficient Frontier
        st.subheader("Efficient Frontier")
        results, weights_record = calculate_efficient_frontier(returns)
        
        max_sharpe_idx = np.argmax(results[2])  # Index of max Sharpe ratio
        sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]  # Standard deviation and return
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(results[0], results[1], c=results[2], cmap='YlGnBu', marker='o')
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(sdp, rp, marker='*', color='r', label='Maximum Sharpe Ratio')
        plt.title('Efficient Frontier')
        plt.xlabel('Risk (Std Deviation)')
        plt.ylabel('Return')
        plt.legend(loc='upper left')
        st.pyplot(plt)
        
        # Risk tolerance and desired return level
        st.subheader("Optimization with Desired Return")
        target_return = st.sidebar.slider("Desired Return Level", min_value=0.0, max_value=1.0, step=0.01)
        optimal_portfolio = portfolio_optimization(returns, target_return)
        
        if optimal_portfolio.success:
            st.write(f"Optimal Portfolio Weights for Desired Return {target_return}:")
            weights_df = pd.DataFrame(optimal_portfolio.x, index=tickers_list, columns=["Weight"])
            st.dataframe(weights_df, width=800, height=400)
            #st.write(weights_df)
        else:
            st.error("Optimization failed to converge.")
    else:
        st.error("Failed to fetch data for the provided tickers.")
        
 