import os
import pandas as pd
pd.set_option('display.max_columns', None)

def read_files(folder_path):
    dataframes = {}

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df_name = file.replace(".csv", "")  # Remove .csv extension for variable name
            file_path = os.path.join(folder_path, file)
            dataframes[df_name] = pd.read_csv(file_path)

    return dataframes