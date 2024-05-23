import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from streamlit_option_menu import option_menu

def normalize_the_price(df):
    """Normalize the prices based on the initial price."""
    historic_data = df.iloc[:, 1:].copy()
    start = df.iloc[0, 1:]
    return historic_data / start

def get_stock_daily_return(df):
    h = df.copy()
    h_shifted = h.shift().fillna(0)
    # Calculate the percentage of change from the previous day
    r = ((h - h_shifted) / h_shifted) * 100
    r.iloc[0] = 0
    return r.astype(np.float64)

def compute_portfolio_return(options, stock_dict, invested_amount):
    stock_dict['SP500'] = "SP500"
    options.append('SP500')

    merged_df = pd.DataFrame()

    # Iterate over the CSV files and stock names
    for stock in options:
        stock_name = stock_dict[stock]
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
    merged_df['SP500'] = merged_df['SP500'].str.replace(',', '').astype(np.float64)
    
    df = merged_df

    df_norm = normalize_the_price(df)

    # These are our random weights
    Weights = np.random.random(len(df.columns[1:]))
    # These Weights should sum to one
    Weights = Weights / sum(Weights)

    # Historic data -- normalized based on initial price
    X = df_norm.values

    portfolio_daily_worth = X @ Weights

    df['portfolio daily worth in $'] = portfolio_daily_worth * invested_amount

    stocks_daily_return = get_stock_daily_return(df_norm)

    # Replace inf and -inf with NaN
    stocks_daily_return.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Replace NaN with the mean of the column
    stocks_daily_return.fillna(stocks_daily_return.mean(), inplace=True)

    beta = {}
    for stock in options:
        if stock_dict[stock] != "SP500":
            security = stocks_daily_return[stock_dict[stock]]
            market = stocks_daily_return['SP500']
            slope, _ = np.polyfit(market, security, deg=1)
            beta[stock_dict[stock]] = slope

    rm = stocks_daily_return['SP500'].mean() * 252

    # Current yield on a U.S. 10-year treasury is 2.5%
    rf = 0.025

    expected_return = {}
    for stock in options:
        if stock_dict[stock] != "SP500":
            er = rf + (beta[stock_dict[stock]] * (rm - rf))
            expected_return[stock_dict[stock]] = er

    weight = 1 / (len(options) - 1)
    weights = weight * np.ones(len(options) - 1)

    expected_return_of_the_portfolio = np.dot(np.array(list(expected_return.values())), weights)

    return df, beta, expected_return, expected_return_of_the_portfolio

def plot_interactive_bars(beta, expected_return):
    beta_fig = go.Figure([go.Bar(x=list(beta.keys()), y=list(beta.values()), name='Beta')])
    beta_fig.update_layout(title='Beta of Stocks', xaxis_title='Stock', yaxis_title='Beta')
    
    expected_return_fig = go.Figure([go.Bar(x=list(expected_return.keys()), y=list(expected_return.values()), name='Expected Return')])
    expected_return_fig.update_layout(title='Expected Return of Stocks', xaxis_title='Stock', yaxis_title='Expected Return')

    return beta_fig, expected_return_fig

def portfolio():
    st.title("Portfolio Return Calculation")

    amount = st.number_input("Amount to Invest", value=100, step=1)

    stock_dict = {
        'Apple Stock (AAPL)': 'AAPL',
        'Boeing (BA)': 'BA',
        'AT&T (T)': 'T',
        'MGM Resorts International (MGM)': 'MGM',
        'Amazon (AMZN)': 'AMZN',
        'IBM': 'IBM',
        'Tesla Motors (TSLA)': 'TSLA',
        'Google (GOOG)': 'GOOG',
    }
    stock_list = list(stock_dict.keys())

    # Create a multiselect dropdown in Streamlit
    selected_options = st.multiselect("Select Assets", stock_list)
    
    button = st.button("Calculate Return")

    if button:
        with st.spinner('Calculating...'):
            df, beta, expected_return, expected_portfolio_return = compute_portfolio_return(selected_options, stock_dict, amount)
            beta_fig, expected_return_fig = plot_interactive_bars(beta, expected_return)

            # Create columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Expected Returns Table")
                expected_return_df = pd.DataFrame.from_dict(expected_return, orient='index', columns=['Expected Return (in %)'])
                st.table(expected_return_df)

            with col2:
                expected_dict = {'Expected Portfolio Return':expected_portfolio_return}
                st.markdown("### Expected Portfolio Return")
                expected_portfolio_return_df = pd.DataFrame.from_dict(expected_dict, orient='index', columns=['Portfolio Return (in %)'])
                st.table(expected_portfolio_return_df)
            

            st.plotly_chart(beta_fig)
            st.plotly_chart(expected_return_fig)

