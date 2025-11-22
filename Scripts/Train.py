#!/usr/bin/env python3

import os
os.path.append("..")
from Library import Model

import pandas as pd

train_df = pd.read_parquet("../Data/train_df.parquet")

Model.train_model(train_df)