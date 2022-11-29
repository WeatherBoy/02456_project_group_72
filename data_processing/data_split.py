import pandas as pd
import numpy as np 


csv_dir = "./data/mr_reddit_casual.csv"
df = pd.read_csv(csv_dir,sep = '§',index_col = False)
n_rows = len(df.index)
split = 0.8

index = np.arange(1,n_rows)
np.random.shuffle(index)

train_num = int(n_rows*(split))
val_num = int(n_rows*(1-split)/2)
test_num = n_rows - train_num - val_num

labels = np.array(['Train']*n_rows)
labels[index[train_num:val_num+train_num]] = 'Val'
labels[index[train_num+val_num:]] = 'Test'


df.loc[:,'split'] = labels
df.to_csv("./data/reddit_casual_split.csv", sep='§',index=False,engine='python')








