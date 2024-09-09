import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Payoff functions
def call_payoff(spot_price, strike_price, is_buy, premium):
    payoff = max(spot_price - strike_price, 0)
    return payoff - premium if is_buy else premium - payoff

def put_payoff(spot_price, strike_price, is_buy, premium):
    payoff = max(strike_price - spot_price, 0)
    return payoff - premium if is_buy else premium - payoff

# Strategy builder page
def strategy_builder_page():
    st.title("Option Strategy Builder")
    st.sidebar.title("Strategy Parameters")

    strategy_type = st.sidebar.selectbox("Select Strategy Type", ["Straddle", "Strangle", "Butterfly", "Bull Spread", "Iron Condor", "Custom"])
    current_price = st.sidebar.number_input("Current Asset Price", value=100.0)

    # Preset strategy builder
    def get_preset_strategy(strategy_name, current_price):
        if strategy_name == "Straddle":
            return [{"type": "Call", "strike": current_price, "is_buy": True, "premium": 5.0},
                    {"type": "Put", "strike": current_price, "is_buy": True, "premium": 5.0}]
        # Add more strategies similarly...
        else:
            return []

    if strategy_type != "Custom":
        options = get_preset_strategy(strategy_type, current_price)
    else:
        num_options = st.sidebar.number_input("Number of Options", min_value=1, max_value=10, value=1)
        options = []
        for i in range(num_options):
            option_type = st.sidebar.selectbox(f"Option Type {i+1}", ["Call", "Put"], key=f"type_{i}")
            strike = st.sidebar.number_input(f"Strike Price {i+1}", value=100.0, key=f"strike_{i}")
            is_buy = st.sidebar.selectbox(f"Buy/Sell {i+1}", ["Buy", "Sell"], key=f"is_buy_{i}")
            premium = st.sidebar.number_input(f"Premium {i+1}", value=5.0, step=0.5, key=f"premium_{i}")
            options.append({"type": option_type, "strike": strike, "is_buy": is_buy == "Buy", "premium": premium})

    # Calculate payoffs
    spot_prices = np.linspace(50, 150, 100)
    total_payoff = np.zeros_like(spot_prices)
    for option in options:
        if option['type'] == "Call":
            payoff = np.array([call_payoff(s, option['strike'], option['is_buy'], option['premium']) for s in spot_prices])
        else:
            payoff = np.array([put_payoff(s, option['strike'], option['is_buy'], option['premium']) for s in spot_prices])
        total_payoff += payoff

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(spot_prices, total_payoff, label="Net Payoff", color="b")
    ax.axhline(0, color='black', lw=1)
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Payoff")
    ax.set_title(f"{strategy_type} Strategy Payoff")
    ax.legend()
    st.pyplot(fig)
