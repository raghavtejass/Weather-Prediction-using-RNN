Team ID ML005
Team Members: Rushyanth S (1CR17CS117)
Raghav S Tejas (1CR17CS119)
Shashank V Sonar (1CR17CS134)
Sugyan Anand Maharana (1CR17CS154)

Abstract
Weather Prediction is notoriously difficult, with numerical methods being the only definitive solution to date, and with large amounts of data to calculate over the duration becomes a challenge. Deep learning strategies have shown an incredible efficiency in predicting periodicity of time series, applying various univariate models based on the Many-to-Many architecture on weather data to contrast long and short-term forecasting between the models for 24 and 72-hour predictions and long term 30 and 60-day predictions, achieving a have achieved an 𝑅2 score of 0.9 for 24-hour short-term forecasting, 0.85 for long-term 72-hour forecasting. Similarly, we have achieved an 𝑅2 score of 0.8 for 60-day long term forecasting and 0.9 for 30-day short term forecasting. It can be concluded that this model can be used for forecasting univariate weather data.

Instructions For Installation of Dependencies

1. Install Pyenv: https://github.com/pyenv/pyenv OR pyenv-win: https://github.com/pyenv-win/pyenv-win
2. Install Poetry: https://python-poetry.org/
3. Ensure Paths for Python are set via the Windows Environments.
4. Using pyenv, do the following: Install python 3.8.1, and rehash:
   1. Command to install python 3.8.1: pyenv install 3.8.1
   2. Command to rehash: pyenv rehash
   3. (If there are errors, refer to the Github Repository for more information)
5. Using Poetry, create a venv inside the repository: poetry config virtual-envs.inproject True
6. Run Installation Command to install dependencies: poetry install


Instructions for Creating Models:
1. Download Dataset.
2. Place Dataset into the folder and rename to dataset.csv
3. Run Datamaker: python Datamaker.py
4. Run Daily Models Generator (Make Sure to edit it to limit the models being generated): python Daily.py
5. Run Hourly Models Generator (Make Sure to edit it to limit the models being generated): python Hourly.py

Instructions for WebFront-End:
1. Run FrontEnd.py: python FrontEnd.py