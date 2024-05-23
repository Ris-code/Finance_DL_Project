import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from Nifty import *
from Stock import *
from portfolio import *
from streamlit_option_menu import option_menu
import base64

# Function to load the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
# Main function to run the Streamlit app
def main():

    # Streamlit Page Configuration
    st.set_page_config(
        page_title="Stock Analyser",
        page_icon="img/img4.png",
        initial_sidebar_state="expanded",
    )

     # Convert image to base64
    img_path = "img/img4.png"
    img_base64 = img_to_base64(img_path)

    # Add logo and text to the sidebar
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{img_base64}" style="width: 60%; height: auto; margin-bottom: 10px;">
            <h1 style="font-size: 40px; margin: 0;">Stock Analyser</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
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

            # Define the relative path to the model
        csv_path_50 = os.path.join(os.path.dirname(__file__), 'Nifty', 'nifty50.csv')
        csv_path_100 = os.path.join(os.path.dirname(__file__), 'Nifty', 'nifty100.csv')
        csv_path_midcap50 = os.path.join(os.path.dirname(__file__), 'Nifty', 'niftymidcap50.csv')

        index_options = {
            "Nifty50": csv_path_50,
            "Nifty100": csv_path_100,
            "Nifty Midcap50": csv_path_midcap50
        }

        file_path = index_options[selected_index]
        df = load_data(file_path)
        # Call the appropriate function based on the selected index
        if selected_index == "Nifty50":
            load_index_data_nifty(df, "50")
        elif selected_index == "Nifty100":
            load_index_data_nifty(df, "100")
        elif selected_index == "Nifty Midcap50":
            load_index_data_nifty(df, "MidCap50")
    elif main_choice == "Stocks":
        stock()
    elif main_choice == "Home":
        st.sidebar.write("Welcome to the Home section.")
    elif main_choice == "Portfolio Management":
        portfolio()

if __name__ == "__main__":
    main()
