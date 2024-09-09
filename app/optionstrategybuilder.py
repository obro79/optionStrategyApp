import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


## cd /Users/owenfisher/Desktop/Machine\ Learning/app/
## streamlit run optionstrategybuilder.py

def display_strategy_builder():
    st.title("Option Strategy Builder")

# Function to calculate the payoff of a Call option
def call_payoff(spot_price, strike_price, is_buy, premium):
    payoff = max(spot_price - strike_price, 0)
    return payoff - premium if is_buy else premium - payoff

# Function to calculate the payoff of a Put option
def put_payoff(spot_price, strike_price, is_buy, premium):
    payoff = max(strike_price - spot_price, 0)
    return payoff - premium if is_buy else premium - payoff

# Function to calculate total payoff of a strategy
def calculate_strategy_payoff(spot_prices, options):
    total_payoff = np.zeros_like(spot_prices)
    for option in options:
        option_type = option['type']
        strike = option['strike']
        is_buy = option['is_buy']
        premium = option['premium']

        if option_type == 'Call':
            payoff = np.array([call_payoff(s, strike, is_buy, premium) for s in spot_prices])
        else:  # Put option
            payoff = np.array([put_payoff(s, strike, is_buy, premium) for s in spot_prices])

        total_payoff += payoff
    
    return total_payoff

# Define preset strategies
def get_preset_strategy(strategy_name, current_price):
    if strategy_name == "Straddle":
        return [
            {"type": "Call", "strike": current_price, "is_buy": True, "premium": 5.0},
            {"type": "Put", "strike": current_price, "is_buy": True, "premium": 5.0}
        ]
    elif strategy_name == "Strangle":
        return [
            {"type": "Call", "strike": current_price + 5, "is_buy": True, "premium": 5.0},
            {"type": "Put", "strike": current_price - 5, "is_buy": True, "premium": 5.0}
        ]
    elif strategy_name == "Butterfly":
        return [
            {"type": "Call", "strike": current_price - 5, "is_buy": True, "premium": 5.0},
            {"type": "Call", "strike": current_price, "is_buy": False, "premium": 5.0},
            {"type": "Call", "strike": current_price + 5, "is_buy": True, "premium": 5.0}
        ]
    elif strategy_name == "Bull Spread":
        return [
            {"type": "Call", "strike": current_price, "is_buy": True, "premium": 5.0},
            {"type": "Call", "strike": current_price + 10, "is_buy": False, "premium": 5.0}
        ]
    elif strategy_name == "Iron Condor":
        return [
            {"type": "Call", "strike": current_price + 10, "is_buy": False, "premium": 5.0},
            {"type": "Call", "strike": current_price + 20, "is_buy": True, "premium": 5.0},
            {"type": "Put", "strike": current_price - 10, "is_buy": False, "premium": 5.0},
            {"type": "Put", "strike": current_price - 20, "is_buy": True, "premium": 5.0}
        ]
    else:  # Custom strategy (no presets)
        return []

# UI Inputs
st.sidebar.title("Option Strategy Builder")

# Option to select a preset strategy or custom
strategy_type = st.sidebar.selectbox("Select Strategy Type", ["Straddle", "Strangle", "Butterfly", "Bull Spread", "Iron Condor", "Custom"])

current_price = st.sidebar.number_input("Current Asset Price", value=100.0)

# Define options based on preset strategy or allow custom inputs
if strategy_type != "Custom":
    options = get_preset_strategy(strategy_type, current_price)
    st.sidebar.write(f"Selected {strategy_type} strategy with the following options:")
    for option in options:
        st.sidebar.write(f"{option['type']} | Strike: {option['strike']} | {'Buy' if option['is_buy'] else 'Sell'} | Premium: {option['premium']}")
else:
    # Custom strategy input
    num_options = st.sidebar.number_input("Number of Options", min_value=1, max_value=10, value=1)
    options = []
    for i in range(num_options):
        st.sidebar.subheader(f"Option {i+1}")
        option_type = st.sidebar.selectbox(f"Option Type {i+1}", ["Call", "Put"], key=f"type_{i}")
        strike = st.sidebar.number_input(f"Strike Price {i+1}", value=100.0, step=1.0, key=f"strike_{i}")
        is_buy = st.sidebar.selectbox(f"Buy/Sell {i+1}", ["Buy", "Sell"], key=f"is_buy_{i}")
        premium = st.sidebar.number_input(f"Premium {i+1}", value=5.0, step=0.5, key=f"premium_{i}")
        options.append({
            "type": option_type,
            "strike": strike,
            "is_buy": is_buy == "Buy",  # Buy = True, Sell = False
            "premium": premium
        })

# Spot price range for the payoff diagram
min_spot_price = st.sidebar.number_input("Min Spot Price", value=50.0)
max_spot_price = st.sidebar.number_input("Max Spot Price", value=150.0)
spot_prices = np.linspace(min_spot_price, max_spot_price, 100)

# Calculate total strategy payoff
strategy_payoff = calculate_strategy_payoff(spot_prices, options)

# Plotting the payoff diagram
st.title(f"{strategy_type} Payoff Diagram")
fig, ax = plt.subplots(figsize=(10, 6))

# Plot individual options in the strategy
for i, option in enumerate(options):
    if option['type'] == 'Call':
        individual_payoff = np.array([call_payoff(s, option['strike'], option['is_buy'], option['premium']) for s in spot_prices])
    else:
        individual_payoff = np.array([put_payoff(s, option['strike'], option['is_buy'], option['premium']) for s in spot_prices])
    
    ax.plot(spot_prices, individual_payoff, label=f"Option {i+1} ({option['type']})")

# Plot net payoff
ax.plot(spot_prices, strategy_payoff, label="Net Payoff", color="b", lw=2)
ax.axhline(0, color='black', lw=1)
ax.set_xlabel("Spot Price")
ax.set_ylabel("Payoff")
ax.set_title(f"{strategy_type} Strategy Payoff")
ax.legend()

st.pyplot(fig)

# Display the options in a table
st.subheader("Options in Strategy")
options_df = {
    "Option Type": [opt["type"] for opt in options],
    "Strike Price": [opt["strike"] for opt in options],
    "Position": ["Buy" if opt["is_buy"] else "Sell" for opt in options],
    "Premium": [opt["premium"] for opt in options]
}
st.table(options_df)

# Profit/Loss Summary
st.subheader("Profit/Loss Summary")
max_profit = np.max(strategy_payoff)
max_loss = np.min(strategy_payoff)
break_even = spot_prices[np.where(np.isclose(strategy_payoff, 0))]

st.write(f"Maximum Profit: ${max_profit:.2f}")
st.write(f"Maximum Loss: ${max_loss:.2f}")
st.write(f"Break-Even Points: {', '.join([f'${be:.2f}' for be in break_even])}")
