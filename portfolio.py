import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.graph_objs as go

def normalize_the_price(df):
  """normalize the prices based on the initial price
  """
  historic_data = df.iloc[:,1:].copy()
  start = df.iloc[0,1:]

  return historic_data/start

def get_stock_daily_return(df):
  h = df.copy()
  h_shifted = h.shift().fillna(0)
  # Calculate the percentage of change from the previous day
  r = ((h - h_shifted)/h_shifted)*100
  r.iloc[0] = 0
  return r.astype(np.float64)

def compute_portfolio_return(options, stock_dict, invested_amount, df_sp500):
    # Iterate over the CSV files and stock names
    for file in stock_dict:
        stock_name = stock_dict[file]
        # Load the CSV file
        df = pd.read_csv(f"Stock/{stock_name}.csv", parse_dates=['Date'], index_col='Date')
        
        # Rename the 'Close' column to the stock name
        df = df[['Close']].rename(columns={'Close': stock_name})
        
        # Merge the DataFrame with the merged_df on the 'Date' column
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = merged_df.join(df, how='outer')

    # Reset the index to make 'Date' a column
    merged_df.reset_index(inplace=True)

    df = merged_df

    df_norm = normalize_the_price(df)

    # These are our random weigths
    Weights = np.random.random(len(df.columns[1:]))
    # These Weights should sum to one
    Weights = Weights/sum(Weights)

    # Historic data -- normalized based on initial price
    X = df_norm.values

    portfolio_daily_worth = X@Weights

    df['portfolio daily worth in $'] = portfolio_daily_worth*invested_amount

    stocks_daily_return = get_stock_daily_return(df_norm)

    beta = {}

    for stock in stock_dict:
        security = stocks_daily_return[stock_dict[stock]]
        market = df_sp500
        slope, y_intercept = np.polyfit(market, security, deg=1) 
        beta[stock]=slope

    plt.bar(beta.keys(),beta.values());
    plt.axhline(y=1.0, c='r');

def portfolio():

    amount = st.number_input("Amount to Invest", value=100, step=1)

    stock_dict = {
        'Apple Stock (AAPL)':'AAPL',
        'Boeing (BA)' : 'BA',
        'AT&T (T)': 'T',
        'MGM Resorts International (MGM)': 'MGM',
        'Amazon (AMZN)' : 'AMZN', 
        'IBM' : 'IBM',
        'Tesla Motors (TSLA)' : 'TSLA',
        'Google (GOOG)' : 'GOOG',
    }
    stock_list = list(stock_dict.keys())

    # Create a multiselect dropdown in Streamlit
    selected_options = st.multiselect("Select Assets", stock_list)
    print(selected_options)

    button = st.button("Calculate Return")

    if button:
        compute_portfolio_return(selected_options, stock_dict, amount, df_sp500)

