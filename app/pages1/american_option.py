import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt, log

# American Option Pricing using Binomial Tree
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

# American Option Pricing page
def american_option_page():
    def apply_page_config():
        st.set_page_config(
            layout="wide",
            initial_sidebar_state="expanded",
            page_title="Option Strategy App"
        ) 
        
    apply_page_config() 
    ##
    
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
    

    st.sidebar.title("American Option Parameters")

    # Sidebar inputs
    current_price = st.sidebar.number_input("Current Asset Price", value=100.0)
    strike_price = st.sidebar.number_input("Strike Price", value=100.0)
    time_to_maturity = st.sidebar.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.sidebar.number_input("Volatility", value=0.2)
    risk_free_interest_rate = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05)
    num_steps = st.sidebar.number_input("Number of Steps", value=5, min_value=1, max_value=50)

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

    
