import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from Model import *
import plotly.express as px
import os

def reformat_date(date):
    return date.replace('-', '/')

def load_data(df):
    df['Date'] = df['Date'].apply(reformat_date)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

def predict_next_5_days(model, last_sequence, scaler):
    next_5_days_predictions = []
    for _ in range(5):
        with torch.no_grad():
            prediction = model(last_sequence)
        next_5_days_predictions.append(scaler.inverse_transform(prediction.numpy()))
        last_sequence = torch.cat([last_sequence[:, 1:, :], prediction.unsqueeze(2)], dim=1)
    return next_5_days_predictions

def display_predictions(dates, predictions):
    box_color = '#0E1117'
    border_color = '#4CAF50'
    st.write('<style>div[data-testid="column"]:nth-of-type(n) {margin-bottom: -35px;}</style>', unsafe_allow_html=True)
    cols = st.columns(len(dates))
    for i, (col, date, price) in enumerate(zip(cols, dates, predictions)):
        col.markdown(
            f"""
            <div style='background-color: {box_color}; padding: 10px; border-radius: 10px; border: 2px solid {border_color}; text-align: center;'>
                <b>{date}</b><br><b>Price:</b> {price[0][0]:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

def plot_candlestick(df):
    df = df.reset_index()
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing=dict(line=dict(color='#2ECC71'), fillcolor='#2ECC71'), 
        decreasing=dict(line=dict(color='#E74C3C'), fillcolor='#E74C3C')
    )])
    fig.update_layout(
        title='Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )
    return fig

def plot_predicted_prices(test_dates, y_test_scaled, predictions, next_5_days_predictions, df):
    last_date = df.index[-1]
    next_5_days_index = pd.date_range(last_date + pd.Timedelta(days=1), periods=5)

    colors = ['red', 'green', 'blue', 'orange', 'purple']
    next_5_days_trace = []
    predict_price = predictions[-1][0]
    for i, price in enumerate(next_5_days_predictions, start=1):
        next_day_trace = go.Scatter(
            x=[last_date, next_5_days_index[i-1]], 
            y=[predict_price, price[0][0]], 
            mode='lines+markers', 
            name=f'Next Day {i}', 
            line=dict(color=colors[i-1]),
            marker=dict(symbol='cross', size=5, color=colors[i-1])
        )
        next_5_days_trace.append(next_day_trace)
        last_date = next_5_days_index[i-1]
        predict_price = price[0][0]

    actual_trace = go.Scatter(x=test_dates, y=y_test_scaled.flatten(), mode='lines', name='Actual Price', line=dict(color='blue'))
    predicted_trace = go.Scatter(x=test_dates, y=predictions.flatten(), mode='lines', name='Predicted Price', line=dict(color='orange'))
    
    traces = [actual_trace, predicted_trace] + next_5_days_trace
    layout = go.Layout(
        title='Actual vs. Predicted Prices with Next 5 Days Predictions',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        legend=dict(orientation='h', yanchor='bottom', y=0.95, xanchor='right', x=1),
        showlegend=True,
        margin=dict(b=80)
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig

def normalize_the_price(df):
    historic_data = df.iloc[:,1:].copy()
    start = df.iloc[0,1:]
    return historic_data/start

def get_stock_daily_return(df):
    h = df.copy()
    h_shifted = h.shift().fillna(0)
    r = ((h - h_shifted)/h_shifted)*100
    r.iloc[0] = 0
    return r.astype(np.float64)

def returns(df, amount_to_invest, stock):
    df = df.reset_index()
    df = df[['Date','Close']]
    df_norm = normalize_the_price(df)
    Weights = np.random.random(len(df.columns[1:]))
    Weights = Weights/sum(Weights)
    X = df_norm.values
    portfolio_daily_worth = X@Weights
    df['portfolio daily worth in $'] = portfolio_daily_worth*amount_to_invest
    stocks_daily_return = get_stock_daily_return(df_norm)
    stocks_daily_return.replace([np.inf, -np.inf], np.nan, inplace=True)
    stocks_daily_return.fillna(stocks_daily_return.mean(), inplace=True)
    date = df['Date']
    df_plot = pd.concat([date, stocks_daily_return], axis=1)
    fig = px.line(df_plot, x='Date', y='Close', title=f'{stock} Returns Over Time')
    return fig

@st.cache_data
def convert_df(df):
    return df.to_csv().encode("utf-8")

def reformat_date_only(date):
    date = str(date)
    return date.split()[0]

def load_index_data(df, index):
    st.title(f"{index} Stock Price Prediction")

    stock_choice = option_menu(
        menu_title="",
        options=["Data", "Candlestick", "Predicted Price", "Returns"],
        icons=["table", "graph-up-arrow", "cash-coin", "cash-stack"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles = {
            "nav-link-selected": {"background-color": "green"},
        }
    )

    model_path = os.path.join(os.path.dirname(__file__), 'Models', f'model_{index}.pth')
    model = load_pytorch_model(model_path)
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.numpy())
    last_sequence = X_test[-1:, :, :]
    next_5_days_predictions = predict_next_5_days(model, last_sequence, scaler)

    # Date Range
    min_date = df.index.min().date()
    max_date = df.index.max().date()

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
    with col2:    
        end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.sidebar.error("Error: End date must fall after start date.")
    else:
        filtered_df = df.loc[start_date:end_date]

        if stock_choice == "Candlestick":
            st.markdown("---")
            st.markdown("### Candlestick Representation")
            st.plotly_chart(plot_candlestick(filtered_df))

        elif stock_choice == "Predicted Price":
            st.markdown("---")
            st.markdown("### Predicted Prices and Next 5 Days Predictions")
            test_dates = filtered_df.index[-len(y_test_scaled):]
            fig = plot_predicted_prices(test_dates, y_test_scaled, predictions, next_5_days_predictions, filtered_df)
            st.plotly_chart(fig)
            st.markdown("### Next 5 Days Predictions")
            next_5_days_index = pd.date_range(filtered_df.index[-1] + pd.Timedelta(days=1), periods=5)
            display_predictions([str(date).split()[0] for date in next_5_days_index], next_5_days_predictions)

        elif stock_choice == "Returns":
            st.markdown("---")
            if 'last_amount' not in st.session_state:
                st.session_state.last_amount = 0

            st.markdown(f"### {index} Stock Returns")
            amount = st.number_input("Amount to Invest", value=100, step=1)
            if st.session_state.last_amount != amount:
                with st.spinner('Loading...'):
                    st.session_state.last_amount = amount
                    fig = returns(filtered_df, amount, index)
                    st.plotly_chart(fig)

        elif stock_choice == "Data":
            st.markdown("---")
            df_1 = filtered_df
            df_1 = df_1.reset_index()
            df_1['Date'] = df_1['Date'].apply(reformat_date_only)
            df_1.set_index('Date', inplace=True)
            data = convert_df(df_1)
            st.download_button(
                label="Download data as CSV",
                data=data,
                file_name="data.csv",
                mime="text/csv",
            )
            st.dataframe(df_1, width=720)

def stock():
    st.title("Stock Price Prediction")

    stock_dict = {
        'Apple Stock (AAPL)':'AAPL',
        'Boeing (BA)' : 'BA',
        'AT&T (T)': 'T',
        'MGM Resorts International (MGM)': 'MGM',
        'Amazon (AMZN)' : 'AMZN', 
        'IBM' : 'IBM',
        'Tesla Motors (TSLA)' : 'TSLA',
        'Google (GOOG)' : 'GOOG',
        'The S&P 500 tracks the performance of 500 major U.S. companies': 'SP500'
    }
    stock_list = list(stock_dict.keys())

    index = st.selectbox(
        "Select the Stock Index",
        stock_list
    )

    if index:
        df = pd.read_csv(f"Stock/{stock_dict[index]}.csv")
        df = load_data(df)
        load_index_data(df, stock_dict[index])

