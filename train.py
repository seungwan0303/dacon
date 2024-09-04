import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost as xgb

train = pd.read_csv("data/images/train.csv")
