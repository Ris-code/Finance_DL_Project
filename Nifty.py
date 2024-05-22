import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from Model import *

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
        # Create the candlestick chart
    df = df.reset_index()
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])

    # Update layout for better visualization
    fig.update_layout(
        title='Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    return fig

# Functions for different indices
def load_index_data(df, index):

    st.title(f"Nifty{index} Price Prediction")

    st.markdown("### Candlestick Representation")
    st.plotly_chart(plot_candlestick(df))

    # Load the pre-trained PyTorch model
    model = load_pytorch_model(f"Models\model_{index}.pth")

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

    # Plotting using Plotly
    test_dates = df.index[-len(y_test_scaled):]

    actual_trace = go.Scatter(x=test_dates, y=y_test_scaled.flatten(), mode='lines', name='Actual Price', line=dict(color='blue'))
    predicted_trace = go.Scatter(x=test_dates, y=predictions.flatten(), mode='lines', name='Predicted Price', line=dict(color='orange'))

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

    traces = [actual_trace, predicted_trace] + next_5_days_trace
    layout = go.Layout(
        title='Actual vs. Predicted Prices with Next 5 Days Predictions',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        legend=dict(orientation='h', yanchor='bottom', y=0.95, xanchor='right', x=1),
        showlegend=True,
        margin=dict(b=80)  # Increase the bottom margin to give more space for the x-axis label
    )

    fig = go.Figure(data=traces, layout=layout)

    # Display the plot
    st.plotly_chart(fig)

    # Display the predictions in separate section
    st.markdown("### Next 5 Days Predictions")
    display_predictions([str(date).split()[0] for date in next_5_days_index], next_5_days_predictions)