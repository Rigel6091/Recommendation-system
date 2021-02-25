import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import TruncatedSVD
import warnings
mov=pd.read_csv('movies.csv')
rat=pd.read_csv('ratings.csv')
mov.head()
rat.head()
df=pd.merge(mov,rat,on="movieId")
df.head()
matrix=df.pivot_table(index='userId',columns='title',values='rating').fillna(0)
X=matrix.values.T
SVD=TruncatedSVD(n_components=20, random_state=0)
matrixp=SVD.fit_transform(X)
matrixp.shape
corr = np.corrcoef(matrixp)
title=matrix.columns
title_list = list(title)
rec = title_list.index('TV Set, The (2006)')
corr_rec  = corr[rec]
list(title[(corr_rec >= 0.9)])
list(set(df["title"]))
