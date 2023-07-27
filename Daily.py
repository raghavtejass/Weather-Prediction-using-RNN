import os


# Suppressing Log Levels to Remove Debug Output From Tensorflow about Cuda.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Base Directory for Models to Be Stored in the Repo, After Training.
MODELDIR = "Models/60DayModels/"

from TestSuite import Models
import pandas as pd

data = pd.read_csv("Data/daily.csv", header=0, index_col=0)

# Types of Networks to be made for comaprison
type = ["Single", "Stacked"]
networks = ["LSTM", "GRU", "BiDirectionalLSTM", "BiDirectionalGRU"]
input_window_size = [90]
output_window_size = [30, 60]

with open("Results_Daily.txt", "w+") as f:
    for ctype in type:
        for cnetworks in networks:
            for cwinp in input_window_size:
                for cwout in output_window_size:

                    # Calling Model Building
                    # IF already built and trained, then the models are simply opened and the result-set is run on them again.
                    model, mse, mae, r2score, history = Models.getResults(
                        # Specify the Base Dir
                        model_base_dir=MODELDIR,
                        # Type of Model to be Trained
                        model_type=ctype + cnetworks,
                        # Data Required to train the Model.
                        data=data,
                        # Number of Ephocs, Can set to any number, patience is set to 10 for maximum gain.
                        epochs=400,
                        # Input Size of the sequences to be trained.
                        input_window_size=cwinp,
                        # Output Size in days of the sequences to be trained.
                        output_window_size=cwout,
                        # The Index of the feature to be trained.
                        feature_idx=0,
                        # Generate Graphs for the results.
                        generate_graph=True,
                    )
                    # Pure Debug Line to check if Training.
                    print("DEBUGLINE:", model, mse, mae, r2score)
                    # Output the Model Score Recieved.
                    f.write(
                        "{} {:.4f} {:.4f} {:.4f}\n".format(model, mse, mae, r2score)
                    )
