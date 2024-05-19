import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.models import model_from_json

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

# Function to predict next 5 days
def predict_next_5_days(model, last_sequence, scaler):
    next_5_days_predictions = []
    for _ in range(5):
        prediction = model.predict(last_sequence)
        next_5_days_predictions.append(scaler.inverse_transform(prediction))
        last_sequence = np.concatenate([last_sequence[:, 1:, :], prediction.reshape(1, 1, 1)], axis=1)
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
    # Load the pre-trained model
    # model_path = "model.h5"
    # model = load_model(model_path)

    # custom_objects = {'Orthogonal': Orthogonal}

    # # Load the model with custom objects
    # model = load_model('model.h5', custom_objects=custom_objects)

    # Load the model from JSON and HDF5
    with open("model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json, custom_objects={'Orthogonal': Orthogonal})
    loaded_model.load_weights("model_weights.h5")

    model = loaded_model

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
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Predicting on the test data
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Inverse transform the y_test to compare with the predictions
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Get the last sequence for predicting next 5 days
    last_sequence = X_test[-1:, :, :]

    # Predict next 5 days
    next_5_days_predictions = predict_next_5_days(model, last_sequence, scaler)

    # Plotting using Plotly
    # Determine the correct index range for actual and predicted values
    test_dates = df.index[-len(y_test):]

    actual_trace = go.Scatter(x=test_dates, y=y_test.flatten(), mode='lines', name='Actual Price', line=dict(color='blue'))
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
