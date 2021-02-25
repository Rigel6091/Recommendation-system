
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import TruncatedSVD
import warnings
In [2]:
mov=pd.read_csv('movies.csv')
In [3]:
rat=pd.read_csv('ratings.csv')
