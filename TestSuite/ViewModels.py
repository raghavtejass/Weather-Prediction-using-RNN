import datetime
import os
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf

from .Suite import get_transformer_with_data


# This function should not try to create model
# but if it does not open, it should throw an error.
def get_pred(
    model_type: str,
    model_base_dir: str,
    pred_time: str,
    data: pd.DataFrame,
    ephocs: int,
    input_size: int,
    output_size: int,
    feature_idx: int,
    start_date: Union[str, datetime.date],
):
    model_name = "{}Model_{}Feature_{}Ephocs_{}In_{}Out".format(
        model_type, feature_idx, ephocs, input_size, output_size
    )

    print(model_name)

    model_path = model_base_dir + model_name

    mmx, normdata = get_transformer_with_data(data, index=feature_idx)

    # Correct the Indexes
    normdata.index = data.index

    middle = pd.to_datetime(start_date)

    if pred_time == "Monthly":
        start = middle - pd.Timedelta(days=input_size - 1)
        end = middle + pd.Timedelta(days=output_size - 1)

    elif pred_time == "Hourly":
        start = middle - pd.Timedelta(hours=input_size - 1)
        end = middle + pd.Timedelta(hours=output_size - 1)

    # print(middle, start, end)

    if not start in normdata.index and not end in normdata.index:
        raise ValueError("Bad Time Stamps")

    # If not bad timestamps
    print(model_path)

    if not os.path.exists(model_path):
        raise ValueError("Model Not Found")

    model = tf.keras.models.load_model(model_path)

    inputs = normdata.loc[start:middle].to_numpy()[np.newaxis, :, :]
    outputs = normdata[middle:end].to_numpy()[np.newaxis, :, :]
    pred = model.predict(inputs)
    pred = pred.reshape(outputs.shape)

    Actual = np.array(mmx.inverse_transform(outputs[0])).flatten()
    Predicted = np.array(mmx.inverse_transform(pred[0])).flatten()

    return Actual, Predicted
