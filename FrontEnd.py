from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import math
import numpy as np
from TestSuite.ViewModels import get_pred


@st.cache
def load_datasets():
    daily = pd.read_csv("Data/daily.csv", header=0, index_col=0)
    daily.index = pd.to_datetime(daily.index)
    hourly = pd.read_csv("Data/hourly.csv", header=0, index_col=0)
    hourly.index = pd.to_datetime(hourly.index)
    return daily, hourly


daily, hourly = load_datasets()

st.write(
    """
# RNN Weather Prediction

Predict Weather using any of the models below and with an input date.
Done By:
Rushyanth S (1CR17CS117)

"""
)

type = st.radio("Select Which Model", ("Stacked", "Single"))

nature = st.radio(
    "Select Which Type", ("LSTM", "GRU", "BiDirectionalLSTM", "BiDirectionalGRU")
)

pred = st.radio("Select Which Prediction", ("Hourly", "Monthly"))

if pred == "Hourly":
    output_size = st.radio("Select how Long to Predict In Hours", (24, 72))
    given = int(output_size)
    input_size = 72
    Ephocs = 100
    ModelDir = "ProdModels/72HourModelsEWM/"
    data = hourly

if pred == "Monthly":
    output_size = st.radio("Select how Long to Predict In Days", (30, 60))
    given = int(output_size)
    input_size = 90
    Ephocs = 80
    ModelDir = "ProdModels/60DayModelsEWM/"
    data = daily

with st.form("Selecting Model"):
    if pred == "Hourly":
        Date = st.date_input(
            "Input a Date to Start Predictions",
            value=hourly.index[90],
            min_value=hourly.index[90],
            max_value=hourly.index[-60],
        )

    if pred == "Monthly":
        Date = st.date_input(
            "Input a Date to Start Predictions",
            value=daily.index[90],
            min_value=daily.index[90],
            max_value=daily.index[-60],
        )

    submitted = st.form_submit_button("Predict Results")

    if submitted:
        real, pred = get_pred(
            model_type=type + nature,
            model_base_dir=ModelDir,
            pred_time=pred,
            data=data,
            ephocs=Ephocs,
            input_size=input_size,
            output_size=output_size,
            feature_idx=0,
            start_date=Date,
        )

        l_label = "Hours" if pred == "Hourly" else "Days"

        ax, fig = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(real, label="Real")
        ax.plot(pred, label="Predicted")
        ax.set_xlabel(l_label)
        ax.set_ylabel("Temp (K)")
        ax.legend()
        ax.set_title("Actual vs Predicted Temperature")

        st.pyplot(fig)
