import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Function to load the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

# Function to create time series dataset
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(50, 25)
        self.fc2 = nn.Linear(25, 1)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # Get the last output of the sequence
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Function to load the PyTorch model
def load_pytorch_model(model_path):
    model = LSTMModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

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

# Functions for different indices
def load_nifty50_data(df):

    st.title("Nifty50 Price Prediction")

    # Load the pre-trained PyTorch model
    model = load_pytorch_model("model.pth")

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
        legend=dict(orientation='h'),
        showlegend=True
    )

    fig = go.Figure(data=traces, layout=layout)

    # Display the plot
    st.plotly_chart(fig)

    # Display the predictions in separate section
    st.markdown("### Next 5 Days Predictions")
    display_predictions([str(date).split()[0] for date in next_5_days_index], next_5_days_predictions)

def load_nifty100_data(df):
    st.write("Loading Nifty100 data...")

def load_nifty_midcap_data(df):
    st.write("Loading Nifty Midcap data...")

# Main function to run the Streamlit app
def main():
    # Set title
    # st.title("Stock Price Prediction App")

    # Sidebar for selecting the index
    index_options = {
        "Nifty50": "nifty50.csv",
        "Nifty100": "nifty100.csv",
        "Nifty Midcap": "nifty_midcap.csv"
    }
    selected_index = st.sidebar.selectbox("Choose an index:", list(index_options.keys()))

    file_path = index_options[selected_index]
    df = load_data(file_path)
    # Call the appropriate function based on the selected index
    if selected_index == "Nifty50":
        load_nifty50_data(df)
    elif selected_index == "Nifty100":
        load_nifty100_data(df)
    elif selected_index == "Nifty Midcap":
        load_nifty_midcap_data(df)

if __name__ == "__main__":
    main()
