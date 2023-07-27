import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as eval_mae
from sklearn.metrics import mean_squared_error as eval_mse
from sklearn.metrics import r2_score as r2
from tensorflow import keras
from tensorflow.keras import callbacks, layers
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from .Suite import generate_test_train_sets, get_transformer_with_data

# This function should:
# Create a Model and Return it
def __get_model(
    input_shape: int,
    output_window_size: int,
    type: str = "SingleLSTM",
):
    model = Sequential()

    if type == "SingleLSTM":
        model.add(layers.LSTM(128, input_shape=input_shape, return_sequences=False))
        model.add(layers.Dense(output_window_size))

    if type == "StackedLSTM":
        model.add(layers.LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.LSTM(64, return_sequences=False))
        model.add(layers.Dense(output_window_size))

    if type == "SingleGRU":
        model.add(layers.GRU(128, input_shape=input_shape, return_sequences=False))
        model.add(layers.Dense(output_window_size))

    if type == "StackedGRU":
        model.add(layers.GRU(128, input_shape=input_shape, return_sequences=True))
        model.add(layers.GRU(128, return_sequences=True))
        model.add(layers.GRU(64, return_sequences=False))
        model.add(layers.Dense(output_window_size))

    if type == "SingleBiDirectionalLSTM":
        model.add(
            layers.Bidirectional(
                layers.LSTM(128, return_sequences=False), input_shape=input_shape
            )
        )
        model.add(output_window_size)

    if type == "StackedBiDirectionalLSTM":
        model.add(
            layers.Bidirectional(
                layers.LSTM(128, return_sequences=True), input_shape=input_shape
            )
        )
        model.add(layers.Bidirectional(layers.LSTM(64), input_shape=input_shape))
        model.add(layers.Dense(output_window_size))

    if type == "SingleBiDirectionalGRU":
        model.add(
            layers.Bidirectional(
                layers.GRU(128, return_sequences=False), input_shape=input_shape
            )
        )
        model.add(output_window_size)

    if type == "StackedBiDirectionalGRU":
        model.add(
            layers.Bidirectional(
                layers.GRU(128, return_sequences=True), input_shape=input_shape
            )
        )
        model.add(layers.Bidirectional(layers.GRU(64), input_shape=input_shape))
        model.add(layers.Dense(output_window_size))

    model.compile(optimizer=Adam(), loss=mean_squared_error)
    return model


def getResults(
    model_type: str,
    model_base_dir: str,
    data: pd.DataFrame,
    epochs: int,
    input_window_size: int,
    output_window_size: int,
    feature_idx: int,
    patience: int = 10,
    generate_graph: bool = False,
):
    model_name = "{}Model_{}Feature_{}Ephocs_{}In_{}Out".format(
        model_type, feature_idx, epochs, input_window_size, output_window_size
    )
    model_path = model_base_dir + model_name

    mmx, normdata = get_transformer_with_data(data=data, index=feature_idx)

    input_shape, x_train, y_train, x_test, y_test = generate_test_train_sets(
        normdata, input_window_size, output_window_size, ratio=0.9
    )


    if not os.path.exists(model_path):
        model = __get_model(
            input_shape=input_shape,
            output_window_size=output_window_size,
        )
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
        model.compile(optimizer=Adam(), loss=mean_squared_error)

        loss = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            verbose=1,
            validation_split=0.1,
            callbacks=[early_stopping],
        )

        model.save(model_path)

        history = loss.history

        with open(model_path + ".pkl", "wb") as f:
            pickle.dump(
                {"val_loss": loss.history["val_loss"], "loss": loss.history["loss"]},
                f,
                pickle.HIGHEST_PROTOCOL,
            )

    else:
        model = keras.models.load_model(model_path)
        with open(model_path + ".pkl", "rb") as f:
            history = pickle.load(f)

    # Predict and Reshape Data.
    predictions = model.predict(x_test, verbose=1)
    predictions = predictions.reshape(y_test.shape)

    real = []
    pred = []
    min_idx_mae, min_mae_score = 0, 99999
    for x in range(predictions.shape[0]):
        a = mmx.inverse_transform(y_test[x])
        real.append(a)

        p = mmx.inverse_transform(predictions[x])
        pred.append(p)

        curr_r2 = eval_mae(
            y_test[x].reshape(1, y_test[x].shape[0]),
            predictions[x].reshape(1, predictions[x].shape[0]),
        )
        if curr_r2 <= min_mae_score:
            min_mae_score = curr_r2
            min_idx_mae = x

    real = np.array(real)
    pred = np.array(pred)

    if generate_graph == True:
        # Calling this to Remove all of the other stuff.
        plt.clf()
        plt.plot(real[min_idx_mae, :, 0], label="Actual")
        plt.plot(pred[min_idx_mae, :, 0], label="Predicted")
        plt.xlabel("Time")
        plt.ylabel(data.columns[feature_idx])
        plt.title("{} Actual Vs Predicted".format(data.columns[feature_idx]))
        plt.legend()
        plt.savefig(model_path + ".svg", format="svg")
        # plt.show(block=True)


    mse = eval_mse(real[:, :, 0], pred[:, :, 0])
    mae = eval_mae(real[:, :, 0], pred[:, :, 0])
    r2score = r2(real[:, :, 0], pred[:, :, 0])

    return model_name, mse, mae, r2score, history
