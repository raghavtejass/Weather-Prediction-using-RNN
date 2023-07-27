import pandas as pd

from TestSuite.Suite import generate_data_suite

data = pd.read_csv("dataset.csv", header=0)

res_dir = "RealData/"

date_time = pd.to_datetime(data.pop("Date Time"), format="%d.%m.%Y %H:%M:%S")
data.index = date_time

selected = ["Tpot (K)"]

removal = [x for x in data.columns if x not in selected]
print(removal)
data = data.drop(labels=removal, axis=1)


def hourly_ewm_mean_data():
    hourly_ewm_mean_data = (
        generate_data_suite(data, "1H", val_choose="mean").ewm(alpha=0.1).mean()
    )
    hourly_ewm_mean_data.to_csv(res_dir + "hourly.csv")


def daily_ewm_mean_data():
    hourly_ewm_mean_data = (
        generate_data_suite(data, "D", val_choose="mean").ewm(alpha=0.1).mean()
    )
    hourly_ewm_mean_data.to_csv(res_dir + "daily.csv")


hourly_ewm_mean_data()
daily_ewm_mean_data()
