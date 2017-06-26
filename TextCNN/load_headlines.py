import pandas
import numpy as np

# Outputs the headlines dataset.
# X is a list of headlines
# y is a list of binary labels, 0 stands for Haaretz and 1 for Israel Hayom
def load_dataset(filenames=['haaretz.csv','israelhayom.csv']):
    cur_y = 0
    X = pandas.DataFrame()
    y = np.empty(0,dtype=np.int32)
    for filename in filenames:
        train_cur = pandas.read_csv(filename, header=None)
        X = pandas.concat([X,train_cur[0]])
        y = np.append(y,cur_y*np.ones(len(train_cur),dtype=np.int32))
        cur_y += 1
    X = [x[0] for x in X.values.tolist()]
    y = y.tolist()
    return X,y
