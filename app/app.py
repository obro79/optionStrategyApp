import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import exp, sqrt, log
from scipy.stats import norm

## TO RUN: streamlit run /Users/owenfisher/Desktop/Machine\ Learning/app/app.py

# Set page config
st.set_page_config(
    layout="wide",  # Use wide layout, and we will center content using CSS
    initial_sidebar_state="expanded"
)

# Add custom CSS for metric displays
st.markdown("""
    <style>
    /* Custom CSS for the metric containers */
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
        font-size: 1.5rem;  /* Adjust font size */
        font-weight: bold;
        margin: 0;
    }

    .metric-label {
        font-size: 1rem;
        margin-bottom: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Display parameters in the main are

# Define the Black-Scholes class for European Options
class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_d1_d2(self, spot_price):
        d1 = (log(spot_price / self.strike) +
              (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (self.volatility * sqrt(self.time_to_maturity))
        d2 = d1 - self.volatility * sqrt(self.time_to_maturity)
        return d1, d2
    
    def calculate_prices(self, spot_price):
        d1, d2 = self.calculate_d1_d2(spot_price)
        call_price = spot_price * norm.cdf(d1) - (self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2))
        put_price = (self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)) - spot_price * norm.cdf(-d1)
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
        
class AmericanOption:
    def __init__(self, spot_price, strike_price, time_to_maturity, volatility, interest_rate, num_steps, option_type="call"):
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.num_steps = num_steps
        self.option_type = option_type

    def price(self):
        dt = self.time_to_maturity / self.num_steps  # Time step
        u = np.exp(self.volatility * sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(self.interest_rate * dt) - d) / (u - d)  # Risk-neutral probability

        # Create the asset price tree
        asset_prices = np.zeros((self.num_steps + 1, self.num_steps + 1))
        for i in range(self.num_steps + 1):
            for j in range(i + 1):
                asset_prices[j, i] = self.spot_price * (u ** (i - j)) * (d ** j)

        # Initialize the option values at maturity
        option_values = np.zeros((self.num_steps + 1, self.num_steps + 1))
        for j in range(self.num_steps + 1):
            if self.option_type == "call":
                option_values[j, self.num_steps] = max(0, asset_prices[j, self.num_steps] - self.strike_price)
            else:  # "put"
                option_values[j, self.num_steps] = max(0, self.strike_price - asset_prices[j, self.num_steps])

        # Backward induction to calculate option price at each node
        for i in range(self.num_steps - 1, -1, -1):
            for j in range(i + 1):
                hold_value = np.exp(-self.interest_rate * dt) * (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1])
                if self.option_type == "call":
                    exercise_value = max(0, asset_prices[j, i] - self.strike_price)
                else:  # "put"
                    exercise_value = max(0, self.strike_price - asset_prices[j, i])
                option_values[j, i] = max(hold_value, exercise_value)  # American option allows early exercise

        return option_values[0, 0]
    
    def visualize_binomial_tree(self):
        dt = self.time_to_maturity / self.num_steps  # Time step
        u = np.exp(self.volatility * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(self.interest_rate * dt) - d) / (u - d)  # Risk-neutral probability
        
        
        #Font adjustment
        font_size = min(6, 12 - self.num_steps // 5)  # Decrease font size as steps increase
    
        annotate_step_interval = 1
        if self.num_steps > 10:
            annotate_step_interval = 5  # Annotate every 5th step
        if self.num_steps > 40:
            annotate_step_interval = 10  # Annotate every 10th step
            
            

    # Create the asset price tree
        asset_prices = np.zeros((self.num_steps + 1, self.num_steps + 1))
        for i in range(self.num_steps + 1):
            for j in range(i + 1):
                asset_prices[j, i] = self.spot_price * (u ** (i - j)) * (d ** j)

    # Plot the binomial tree as a heatmap using matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Binomial Tree of Asset Prices ({self.num_steps} Steps)")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Price Levels")
        ax.tick_params(axis='both', which='major', labelsize=font_size + 2)
        ax.tick_params(axis='both', which='minor', labelsize=font_size + 4)


    # Annotate the plot with asset prices
        for i in range(0, self.num_steps + 1, annotate_step_interval):
            for j in range(i+1):
                ax.text(i, j, f'{asset_prices[j, i]:.2f}', ha='center', va='center', fontsize=font_size, color='black')
    
    
    # Plot a heatmap-like background using the data
        
        cax = ax.imshow(asset_prices, cmap="YlGnBu", interpolation='nearest', aspect='auto', origin='lower')
        fig.colorbar(cax, ax=ax, label="Asset Price")

        ax.set_xticks(np.arange(self.num_steps + 1))
        ax.set_yticks(np.arange(self.num_steps + 1))

        st.pyplot(fig)

# Sidebar inputs
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None
    
col1, col2 = st.sidebar.columns(2)

# Add buttons to the columns
with col1:
    if st.button("American Option"):
        st.session_state.selected_option = "american"

with col2:
    if st.button("European Option"):
        st.session_state.selected_option = "european"
                
current_price = st.sidebar.number_input("Current Asset Price", value=100.0, key="1")
strike_price = st.sidebar.number_input("Strike Price", value=100.0)
time_to_maturity = st.sidebar.number_input("Time to Maturity (Years)", value=2.0)
volatility = st.sidebar.number_input("Volatility", value=0.2)
risk_free_interest_rate = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05)

# Display parameters in the main area

if st.session_state.selected_option == "american":
    # Use Binomial Tree model for American Options
    num_steps = st.sidebar.number_input("Number of Steps (for American Option)", value=5, min_value=1, step=1, max_value=45)


    # Initialize and calculate the American option price
    american_call_option = AmericanOption(current_price, strike_price, time_to_maturity, volatility, risk_free_interest_rate, num_steps, "call")
    american_put_option = AmericanOption(current_price, strike_price, time_to_maturity, volatility, risk_free_interest_rate, num_steps, "put")
    
    american_call_price = american_call_option.price()
    american_put_price = american_put_option.price()
    
    # Display the title for the American option pricing model
    st.title("American Option Pricing Model")
    
    # Display the American option prices above the binomial tree
    col1, col2 = st.columns([1, 1], gap="small")
    with col1:
        st.markdown(f"""
            <div class="metric-container metric-call">
                <div>
                    <div class="metric-label">{"CALL"} Value</div>
                    <div class="metric-value">${american_call_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        american_call_option.visualize_binomial_tree()
    
    with col2:
        st.markdown(f"""
            <div class="metric-container metric-put">
                <div>
                    <div class="metric-label">{"PUT"} Value</div>
                    <div class="metric-value">${american_put_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        american_put_option.visualize_binomial_tree()   

elif st.session_state.selected_option == "european":
    # Heatmap parameters for European options
    st.title("European Black Scholes Pricing Model")
    st.sidebar.title("Heat Map Parameters")
    min_spot_price = st.sidebar.number_input("Min Spot Price", value=80.0)
    max_spot_price = st.sidebar.number_input("Max Spot Price", value=120.0)
    min_volatility_for_heatmap = st.sidebar.slider("Min Volatility for Heatmap", min_value=0.1, max_value=1.0, value=0.2, step=0.01)
    max_volatility_for_heatmap = st.sidebar.slider("Max Volatility for Heatmap", min_value=0.1, max_value=1.0, value=0.8, step=0.01)

    bs_model = BlackScholes(time_to_maturity, strike_price, current_price, volatility, risk_free_interest_rate)
    call_price, put_price = bs_model.calculate_prices(current_price)
    
    call_greeks, put_greeks = bs_model.calculate_greeks(current_price)
    call_greeks_df = pd.DataFrame(call_greeks)
    put_greeks_df = pd.DataFrame(put_greeks)
    
    col1, col2 = st.columns([1, 1], gap="small")
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
        fig_call, ax_call = plt.subplots(figsize=(20, 16))  # Increase figsize for bigger heatmap
        sns.heatmap(call_prices, xticklabels=np.round(spot_prices, 2), yticklabels=np.round(volatilities, 2),
                    annot=True, fmt=".2f", cmap="viridis", ax=ax_call, annot_kws={"size": 16}, cbar_kws={"label": "Call Price"})
        ax_call.set_xlabel('Spot Price')
        ax_call.set_ylabel('Volatility')
        st.pyplot(fig_call)
        st.markdown("<h3 style='text-align: center;'>Call Greeks</h3>", unsafe_allow_html=True)
        st.markdown(call_greeks_df.to_html(index=False), unsafe_allow_html=True)
        

    with col4:
        st.subheader("Put Price Heatmap")
        fig_put, ax_put = plt.subplots(figsize=(20, 16))  # Increase figsize for bigger heatmap
        sns.heatmap(put_prices, xticklabels=np.round(spot_prices, 2), yticklabels=np.round(volatilities, 2), annot=True, fmt=".2f",
                    cmap="viridis", ax=ax_put, annot_kws={"size": 16})
        ax_put.set_xlabel('Spot Price')
        ax_put.set_ylabel('Volatility')
        st.pyplot(fig_put)
        st.markdown("<h3 style='text-align: center;'>Put Greeks</h3>", unsafe_allow_html=True)
        st.markdown(put_greeks_df.to_html(index=False), unsafe_allow_html=True)

elif st.session_state.selected_option == 'strategy_builder':   
    display_strategy_builder()     
    
else:
    st.write("Please select an option type to calculate the price (American or European).")

st.markdown("""
    <style>
    .center-table {
        display: flex;
        justify-content: center;
    }
    table {
        width: 600px;  /* Adjust the width of the tables */
        margin: 0 auto;
        font-size: 24px;  /* Adjust font size */
    }
    th, td {
        text-align: center;
        padding: 10px;  /* Add padding to cells */
    }
    </style>
""", unsafe_allow_html=True)


    



