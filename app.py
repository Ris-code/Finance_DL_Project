import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from Nifty import *
from streamlit_option_menu import option_menu

# Function to load the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

# Main function to run the Streamlit app
def main():
    # Sidebar with navigation menu
    with st.sidebar:
        main_choice = option_menu(
            menu_title="",
            options=["Home", "Stocks", "Nifty", "Portfolio Management"],
            icons=["house", "graph-up", "bar-chart", "briefcase"],
            menu_icon="cast",
            default_index=0,
        )

    if main_choice == "Nifty":
        # Sidebar for selecting the index
        with st.sidebar:
            selected_index = option_menu(
                menu_title="Choose index:",
                options=["Nifty50", "Nifty100", "Nifty Midcap50"],
                icons=["list", "list", "list"],
                menu_icon="cast",
                default_index=0,
            )

        index_options = {
            "Nifty50": r"Nifty\nifty50.csv",
            "Nifty100": r"Nifty\nifty100.csv",
            "Nifty Midcap50": r"Nifty\niftymidcap50.csv"
        }

        file_path = index_options[selected_index]
        df = load_data(file_path)
        # Call the appropriate function based on the selected index
        if selected_index == "Nifty50":
            load_index_data(df, "50")
        elif selected_index == "Nifty100":
            load_index_data(df, "100")
        elif selected_index == "Nifty Midcap50":
            load_index_data(df, "MidCap50")
    elif main_choice == "Stocks":
        st.sidebar.write("Stocks section not implemented yet.")
    elif main_choice == "Home":
        st.sidebar.write("Welcome to the Home section.")

if __name__ == "__main__":
    main()
