import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from datetime import datetime


def plot_predictions(db_url="sqlite:///pizza_sales.db"):
    engine = create_engine(db_url)

    final_predictions = pd.read_sql('SELECT * FROM final_predictions', engine)

    y_pred = final_predictions['y_pred']
    y_true = final_predictions['y_true']
    time = final_predictions['datetime']

    # Plot the results
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time, y_pred, label='Prediction', color='darkorange')
    ax.plot(time, y_true, label='True', color='dodgerblue')
    ax.set_title('The Hawaiian sold quantities: True vs Predicted (30 days into the future)', fontsize=16)
    ax.set_ylabel('Quantity', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    plt.legend()
    plt.show()


st.title("Model Predictions vs Real Data")

# Assuming you're okay with the default database URL
db_url = "sqlite:///pizza_sales.db"  # Or use st.text_input to get user input

plot = plot_predictions(db_url)
st.pyplot(plot)
