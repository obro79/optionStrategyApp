import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


#TODO add dynamic way to handle lots of simulations/ give warning.
def monte_carlo_page():
    st.set_page_config(layout="wide",initial_sidebar_state="expanded") 
        
    st.title("Monte Carlo Simulation for Option Pricing")

    # Sidebar for input parameters
    st.sidebar.title("Simulation Parameters")
    
    current_price = st.sidebar.number_input("Current Asset Price", value=100.0)
    strike_price = st.sidebar.number_input("Strike Price", value=100.0)
    time_to_maturity = st.sidebar.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.sidebar.number_input("Volatility (σ)", value=0.2)
    risk_free_rate = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05)
    num_simulations = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=100000, value=1000, step=1000)
    num_steps = st.sidebar.number_input("Number of Steps", min_value=50, max_value=504, value=252, step=50)

    option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])

    # Checkbox to compare with Black-Scholes
    compare_bs = st.sidebar.checkbox("Compare with Black-Scholes Model")

    # Function to calculate Black-Scholes price
    def black_scholes_price(current_price, strike_price, time_to_maturity, volatility, risk_free_rate, option_type):
        d1 = (np.log(current_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
        d2 = d1 - volatility * np.sqrt(time_to_maturity)
        
        if option_type == "Call":
            price = current_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
        else:
            price = strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2) - current_price * norm.cdf(-d1)
        
        return price
    
    # Function to run Monte Carlo simulation
    
    def monte_carlo_simulation_paths(current_price, time_to_maturity, volatility, risk_free_rate, num_simulations, num_steps):
        dt = time_to_maturity / num_steps
        price_paths = np.zeros((num_steps, num_simulations))
        price_paths[0] = current_price

        for t in range(1, num_steps):
            z = np.random.standard_normal(num_simulations)
            price_paths[t] = price_paths[t - 1] * np.exp((risk_free_rate - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * z)

        return price_paths
    
    def monte_carlo_simulation(current_price, strike_price, time_to_maturity, volatility, risk_free_rate, num_simulations, num_steps, option_type):
        dt = time_to_maturity / num_steps
        payoff_sum = 0.0
        final_prices = []

        for i in range(num_simulations):
            # Generate a random price path
            prices = np.zeros(num_steps)
            prices[0] = current_price
            for t in range(1, num_steps):
                random_shock = np.random.normal(0, 1)
                prices[t] = prices[t - 1] * np.exp((risk_free_rate - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * random_shock)
            final_prices.append(prices[-1])

            # Calculate the payoff at maturity
            if option_type == "Call":
                payoff = max(prices[-1] - strike_price, 0)
            else:
                payoff = max(strike_price - prices[-1], 0)
            
            payoff_sum += payoff
        
        # Discount the average payoff to get the present value
        discounted_price = np.exp(-risk_free_rate * time_to_maturity) * (payoff_sum / num_simulations)
        return discounted_price, np.array(final_prices)
    
    # Run the Monte Carlo simulation
    monte_carlo_price, final_prices = monte_carlo_simulation(
        current_price, strike_price, time_to_maturity, volatility, risk_free_rate, num_simulations, num_steps, option_type
    )
    
    price_paths = monte_carlo_simulation_paths(
        current_price, time_to_maturity, volatility, risk_free_rate, num_simulations, num_steps
    )

    # Visualize the simulation paths
    st.subheader("Simulated Stock Price Paths")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each path
    time_grid = np.linspace(0, time_to_maturity, num_steps)
    for i in range(min(num_simulations, 100)):  # Plot only up to 100 paths for better visualization
        ax.plot(time_grid, price_paths[:, i], lw=1, alpha=0.7)

    ax.set_title("Simulated Stock Price Paths")
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Price")
    st.pyplot(fig)
    
    # Calculate the Black-Scholes price for comparison TODO stylize the text here
    if compare_bs:
        bs_price = black_scholes_price(current_price, strike_price, time_to_maturity, volatility, risk_free_rate, option_type)
        st.write(f"Black-Scholes {option_type} Option Price: ${bs_price:.2f}")
    
    # Display the results
    st.write(f"Monte Carlo {option_type} Option Price: ${monte_carlo_price:.2f}")
    
    # Visualize the final prices
    st.subheader("Distribution of Final Simulated Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(final_prices, bins=50, alpha=0.7, color='blue')
    ax.set_title("Histogram of Final Asset Prices")
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

monte_carlo_page()
