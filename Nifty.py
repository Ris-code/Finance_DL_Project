import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from Model import *
from streamlit_option_menu import option_menu
import os

# Function to create time series dataset
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Function to predict next 5 days
def predict_next_5_days(model, last_sequence, scaler):
    next_5_days_predictions = []
    for _ in range(5):
        with torch.no_grad():
            prediction = model(last_sequence)
        next_5_days_predictions.append(scaler.inverse_transform(prediction.numpy()))
        last_sequence = torch.cat([last_sequence[:, 1:, :], prediction.unsqueeze(2)], dim=1)
    return next_5_days_predictions

# Function to display date and price in colored boxes
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
        close=df['Close']
    )])
    fig.update_layout(
        title='Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )
    return fig

def reformat_date(date):
    date = str(date)
    return date.split()[0]

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

@st.cache_data
def convert_df(df):
    return df.to_csv().encode("utf-8")

# Functions for different indices
def load_index_data_nifty(df, index):
    st.title(f"Nifty {index} Price Prediction")

    # with st.expander("Menu", expanded=True):
    choice = option_menu(
        menu_title="",
        options=["Data", "Candlestick", "Predicted Price"],
        icons=["table", "graph-up-arrow", "cash-coin"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles = {
            "nav-link-selected": {"background-color": "green"},
        }
    )

    # Load the pre-trained PyTorch model
    # model = load_pytorch_model(f"Models/model_{index}.pth")
    # Define the relative path to the model
    model_path = os.path.join(os.path.dirname(__file__), 'Models', f'model_{index}.pth')

    # Load the pre-trained PyTorch model
    # model = load_pytorch_model(f"Models\model_{index}.pth")
    model = load_pytorch_model(model_path)

    # Selecting the feature and target columns
    data = df[['Close']].values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Split data into train and test sets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Create time series data
    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input to be [samples, time steps, features] required for LSTM
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Predicting on the test data
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    predictions = scaler.inverse_transform(predictions)

    # Inverse transform the y_test to compare with the predictions
    y_test_scaled = scaler.inverse_transform(y_test.numpy())

    # Get the last sequence for predicting next 5 days
    last_sequence = X_test[-1:, :, :]

    # Predict next 5 days
    next_5_days_predictions = predict_next_5_days(model, last_sequence, scaler)

    # Plotting based on the clicked button
    if choice == "Candlestick":
        st.markdown("### Candlestick Representation")
        st.plotly_chart(plot_candlestick(df))

    elif choice == "Predicted Price":
        st.markdown("### Predicted Prices and Next 5 Days Predictions")
        test_dates = df.index[-len(y_test_scaled):]
        fig = plot_predicted_prices(test_dates, y_test_scaled, predictions, next_5_days_predictions, df)
        st.plotly_chart(fig)
        st.markdown("### Next 5 Days Predictions")
        next_5_days_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=5)
        display_predictions([str(date).split()[0] for date in next_5_days_index], next_5_days_predictions)
    
    elif choice == "Data":
        df_1 = df
        df_1 = df_1.reset_index()
        df_1['Date'] = df_1['Date'].apply(reformat_date)
        df_1.set_index('Date', inplace=True)

        data = convert_df(df_1)

        st.download_button(
            label="Download data as CSV",
            data=data,
            file_name="data.csv",
            mime="text/csv",
        )

        st.dataframe(df_1, width=720)
