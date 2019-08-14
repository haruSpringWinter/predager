import numpy as np
import pandas as pd

def parse(path):
    df = pd.read_csv(path)
    return np.array(df)
