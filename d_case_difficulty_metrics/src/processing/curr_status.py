import os
import sys
import pandas as pd


def curr_status(n_samples, file_name):
    if not os.path.isfile(file_name):
        print("File does not exist, starting from index 0.")
        curr_df = pd.DataFrame()
        return 0, curr_df
    try:
        curr_df = pd.read_excel(file_name)

    except Exception:
        print("Error reading the file")
        sys.exit(0)

    starting = len(curr_df)
    if starting == 0:
        print("Starting index: 0")
        return 0, curr_df

    if starting == n_samples:
        print("All samples are done.")
        return starting, curr_df

    print("Starting index:", starting)
    return starting, curr_df
