import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import os
print('current path', os.getcwd())
data = pd.read_csv('e76d9c91-05d4-4307-a9fa-d7ae79884931_Diabetess.csv')

