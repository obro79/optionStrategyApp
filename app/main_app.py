import streamlit as st

# Import the necessary pages
from pages1.european_option import european_option_page
from pages1.american_option import american_option_page
from pages1.option_strategy import option_strategy_page
from pages1.monte_carlo import monte_carlo_page
from pages1.portfolio import portfolio_management_page

## source myenv/bin/activate
## cd /Users/owenfisher/Desktop/optionStrategyApp/app
##streamlit run main_app.py

# Navigation for the app
pages = {
    "Options": [
        st.Page(european_option_page, title="European Option", icon="📈"),
        st.Page(american_option_page, title="American Option", icon="📊"),
        st.Page(monte_carlo_page, title="Monte Carlo Simulation", icon="📉"), # change icon later
    ],
    "Portfolio Managing": [
        st.Page(portfolio_management_page, title="Portfolio Optimizer", icon="〽️") #TODO icon
    ],
    "Strategies": [
        st.Page(option_strategy_page, title="Option Strategy Builder", icon="🛠️")
    ]
}

# Show the navigation in the sidebar
pg = st.navigation(pages, position="sidebar")

# Run the current page selected by the user
pg.run()
##

