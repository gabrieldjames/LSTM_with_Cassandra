import pandas as pd
from comment_generation import run_lstm


df = pd.read_csv('char_data.csv')

run_lstm(df, 40, 3, 5, train = True, save = True)
