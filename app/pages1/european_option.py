import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy import exp, sqrt, log

# Define the Black-Scholes class
def european_option_page():
    def apply_page_config():
        st.set_page_config(
            layout="wide",
            initial_sidebar_state="expanded",
            page_title="Option Strategy App"
        ) 
        
    apply_page_config() 
           
    st.markdown("""
            <style>
        /* Center the main content */
        .center-container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            width: 100%;
        }

        /* Style for heatmap sections */
        .heatmap-container {
            background-color: #1f2937;  /* Subtle grayish background */
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;  /* Add space between heatmaps and other content */
        }

        /* Style for metric containers (CALL and PUT values) */
        .metric-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 8px;
            width: auto;
            margin: 0 auto;
        }

        .metric-call {
            background-color: #90ee90;  /* Light green background */
            color: black;
            margin-right: 10px;
            border-radius: 10px;
        }

        .metric-put {
            background-color: #ffcccb;  /* Light red background */
            color: black;
            border-radius: 10px;
        }

        .metric-value {
            font-size: 2rem;  /* Increase font size */
            font-weight: bold;
            margin: 0;
        }

        .metric-label {
            font-size: 1.5rem;  /* Increase font size */
            margin-bottom: 4px;
        }

        /* Style for tables (Greeks) */
        .center-table {
            display: flex;
            justify-content: center;
        }

        table {
            width: 700px;  /* Adjust the width of the tables */
            margin: 0 auto;
            font-size: 18px;  /* Adjust font size */
        }

        th, td {
            text-align: center;
            padding: 10px;  /* Add padding to cells */
        }
        </style>

            
        """, unsafe_allow_html=True)
      
    class BlackScholes:
        def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
            self.time_to_maturity = time_to_maturity
            self.strike = strike
            self.current_price = current_price
            self.volatility = volatility
            self.interest_rate = interest_rate

        def calculate_d1_d2(self, spot_price):
            d1 = (np.log(spot_price / self.strike) +
                 (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (self.volatility * np.sqrt(self.time_to_maturity))
            d2 = d1 - self.volatility * np.sqrt(self.time_to_maturity)
            return d1, d2

        def calculate_prices(self, spot_price):
            d1, d2 = self.calculate_d1_d2(spot_price)
            call_price = spot_price * norm.cdf(d1) - (self.strike * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2))
            put_price = (self.strike * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)) - spot_price * norm.cdf(-d1)
            return call_price, put_price
        
        def calculate_greeks(self, spot_price):
            d1, d2 = self.calculate_d1_d2(spot_price)
            # Call Greeks
            call_delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (self.current_price * self.volatility * sqrt(self.time_to_maturity))
            vega = self.current_price * norm.pdf(d1) * sqrt(self.time_to_maturity)
            call_theta = (-self.current_price * norm.pdf(d1) * self.volatility / (2 * sqrt(self.time_to_maturity))) - \
                        (self.interest_rate * self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2))
            call_rho = self.strike * self.time_to_maturity * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)

            # Put Greeks
            put_delta = call_delta - 1
            put_theta = (-self.current_price * norm.pdf(d1) * self.volatility / (2 * sqrt(self.time_to_maturity))) + \
                        (self.interest_rate * self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2))
            put_rho = -self.strike * self.time_to_maturity * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)

            # Return the Greeks as dictionaries
            call_greeks = {
                'Greek': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
                'Value': [round(call_delta, 4), round(gamma, 4), round(vega,4), round(call_theta, 4), round(call_rho,4)]
            }

            put_greeks = {
                'Greek': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
                'Value': [round(put_delta , 4), round(gamma, 4), round(vega,4), round(put_theta, 4) , round(put_rho, 4)]
            }

            return call_greeks, put_greeks

# UI Sidebar inputs
    st.sidebar.title("European Option Parameters")
    current_price = st.sidebar.number_input("Current Asset Price", value=100.0)
    strike_price = st.sidebar.number_input("Strike Price", value=100.0)
    time_to_maturity = st.sidebar.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.sidebar.number_input("Volatility", value=0.20)
    risk_free_interest_rate = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05)

# Black-Scholes model instantiation
    bs_model = BlackScholes(time_to_maturity, strike_price, current_price, volatility, risk_free_interest_rate)
    call_price, put_price = bs_model.calculate_prices(current_price)

# Display the prices and Greeks
    st.title("European Black-Scholes Pricing Model")

# Heatmap parameters
    
    call_greeks, put_greeks = bs_model.calculate_greeks(current_price)
    call_greeks_df = pd.DataFrame(call_greeks)
    put_greeks_df = pd.DataFrame(put_greeks)
    
    st.markdown("""
    <style>
    /* Remove extra padding */
    .css-1lcbmhc.e1fqkh3o3 {
        padding: 0rem 0rem 0rem 0rem;
    }

    /* Expand the heatmaps and columns */
    .st-bh {
        margin: 0 auto;
    }

    .st-af {
        display: flex;
        justify-content: space-between;
        margin: 0 auto;
    }

    /* Make heatmaps full width */
    .full-width {
        width: 100%;
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Call Price Metric for European Option
        st.markdown(f"""
            <div class="metric-container metric-call">
                <div>
                    <div class="metric-label">CALL Value</div>
                    <div class="metric-value">${call_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Put Price Metric for European Option
        st.markdown(f"""
            <div class="metric-container metric-put">
                <div>
                    <div class="metric-label">PUT Value</div>
                    <div class="metric-value">${put_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    # Generate spot prices and volatilities for the heatmap
    st.sidebar.title("Heat Map Parameters")
    min_spot_price = st.sidebar.number_input("Min Spot Price", value=80.0)
    max_spot_price = st.sidebar.number_input("Max Spot Price", value=120.0)
    min_volatility_for_heatmap = st.sidebar.slider("Min Volatility for Heatmap", min_value=0.1, max_value=1.0, value=0.2, step=0.01)
    max_volatility_for_heatmap = st.sidebar.slider("Max Volatility for Heatmap", min_value=0.1, max_value=1.0, value=0.8, step=0.01)
    spot_prices = np.linspace(min_spot_price, max_spot_price, 10)
    volatilities = np.linspace(min_volatility_for_heatmap, max_volatility_for_heatmap, 10)

    # Initialize arrays for call and put prices
    call_prices = np.zeros((len(volatilities), len(spot_prices)))
    put_prices = np.zeros((len(volatilities), len(spot_prices)))

    # Calculate call and put prices for each combination of spot price and volatility
    for i, vol in enumerate(volatilities):
        for j, spot in enumerate(spot_prices):
        # Create a new Black-Scholes model with varying volatilities
            bs_model = BlackScholes(time_to_maturity, strike_price, spot, vol, risk_free_interest_rate)
            call_prices[i, j], put_prices[i, j] = bs_model.calculate_prices(spot)

    # Display heatmaps for the European options
    st.subheader("European Option Pricing - Interactive Heat Map")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Call Price Heatmap")
        fig_call, ax_call = plt.subplots(figsize=(24, 18))  # Increase figsize for bigger heatmap
        sns.heatmap(call_prices, xticklabels=np.round(spot_prices, 1), yticklabels=np.round(volatilities, 1),
                    annot=True, fmt=".2f", cmap="viridis", ax=ax_call, annot_kws={"size": 24}, cbar_kws={"label": "Call Price"})
        ax_call.set_xlabel('Spot Price', fontsize=24)
        ax_call.set_ylabel('Volatility', fontsize=24)
        ax_call.tick_params(axis='both', which='major', labelsize=24)
        st.pyplot(fig_call)
        st.markdown("<h3 style='text-align: center;'>Call Greeks</h3>", unsafe_allow_html=True)
        st.markdown(call_greeks_df.to_html(index=False), unsafe_allow_html=True)
        
    with col4:
        st.subheader("Put Price Heatmap")
        fig_put, ax_put = plt.subplots(figsize=(24, 18))  # Increase figsize for bigger heatmap
        sns.heatmap(put_prices, xticklabels=np.round(spot_prices, 1), yticklabels=np.round(volatilities, 1), annot=True, fmt=".2f",
                    cmap="viridis", ax=ax_put, annot_kws={"size": 24})
        ax_put.set_xlabel('Spot Price', fontsize=24)
        ax_put.set_ylabel('Volatility', fontsize=24)
        ax_put.tick_params(axis='both', which='major', labelsize=24)
        
        st.pyplot(fig_put)
        st.markdown("<h3 style='text-align: center;'>Put Greeks</h3>", unsafe_allow_html=True)
        st.markdown(put_greeks_df.to_html(index=False), unsafe_allow_html=True)

    st.markdown("""
        <style>
        .center-table {
            display: flex;
            justify-content: center;
        }
        table {
            width: 100%;  /* Adjust the width of the tables */
            margin: 0 auto;
            font-size: 24px;  /* Adjust font size */
        }
        th, td {
            text-align: center;
            padding: 10px;  /* Add padding to cells */
        }
        </style>
    """, unsafe_allow_html=True)