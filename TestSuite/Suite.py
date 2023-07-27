from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Split Dataset into Train and Test Sets.
def split_data(data: pd.DataFrame, ratio: int = 0.6):
    # Split the data into Test and Train.
    train_end = int(len(data) * ratio)

    train_data = data.iloc[:train_end, :]
    test_data = data.iloc[train_end:, :]

    return train_data, test_data


def adv_rolling_window(
    data: pd.DataFrame,
    feature_idx: int = 0,
    input_size: int = 30,
    output_size: int = 30,
) -> Tuple[np.array, np.array]:

    # Extract the useable data,
    # Reshape it.
    input_data = data.iloc[:, feature_idx].to_numpy().reshape(-1, 1)

    X, Y = list(), list()

    # The end_range should be less than the maximum length
    # which is given by
    # end - (output_size)
    end_range = len(input_data) - (output_size + input_size - 1)

    # For Each Index, We do the following,
    # Sliding Window Approach
    for idx in range(0, end_range):
        # From (idx, idx+inp)
        x_start = idx
        x_end = idx + input_size

        # from (idx+inp, idx+inp+out) - overlap
        y_start = idx + input_size
        y_end = idx + input_size + output_size

        x_data = input_data[x_start:x_end, :]
        y_data = input_data[y_start:y_end, :]

        X.append(x_data)
        Y.append(y_data)

    return np.array(X), np.array(Y)


def get_transformer_with_data(data: pd.DataFrame, index: int = 0, mmx=MinMaxScaler()):

    # Reshape For a Single Feature.
    req_data = data.iloc[:, index].to_numpy().reshape(-1, 1)

    scaled = mmx.fit_transform(req_data)
    normdata = pd.DataFrame(scaled, columns=[data.columns[index]])

    return mmx, normdata


def generate_data_suite(data: pd.DataFrame, time_period: str, val_choose="max"):
    resampled_data = data.resample(time_period)
    if val_choose == "max":
        resampled_data = resampled_data.max()
    else:
        resampled_data = resampled_data.mean()

    resampled_data = resampled_data.fillna(method="backfill")

    return resampled_data


def generate_test_train_sets(
    data: pd.DataFrame,
    input_window_size: int,
    output_window_size: int,
    ratio: int = 0.9,
):
    # Feature will always
    input_shape = (input_window_size, 1)

    train, test = split_data(data, ratio=ratio)

    x_train, y_train = adv_rolling_window(
        train,
        input_size=input_window_size,
        output_size=output_window_size,
    )

    x_test, y_test = adv_rolling_window(
        test,
        input_size=input_window_size,
        output_size=output_window_size,
    )

    return input_shape, x_train, y_train, x_test, y_test
