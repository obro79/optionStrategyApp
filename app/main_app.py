import streamlit as st

# Import the necessary pages
from pages1.european_option import european_option_page
from pages1.american_option import american_option_page
from pages1.strategy_builder import strategy_builder_page
from pages1.option_strategy import option_strategy_page

## source myenv/bin/activate
## cd /Users/owenfisher/Desktop/optionStrategyApp/app
##streamlit run main_app.py


# Navigation for the app
pages = {
    "Options": [
        st.Page(european_option_page, title="European Option", icon="📈"),
        st.Page(american_option_page, title="American Option", icon="📊")
    ],
    "Strategies": [
        st.Page(option_strategy_page, title="Option Strategy Builder", icon="🛠️")
    ]
}

# Show the navigation in the sidebar
pg = st.navigation(pages, position="sidebar")

# Run the current page selected by the user
pg.run()

