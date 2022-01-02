import numpy as np
import pandas as pd

MODULO = 232792560 # divided by 2, 3, 4, ..., 22

# modulo using constant value MODULO
# result range : - (MODULO / 2) ~ + (MODULO / 2)
def modulo(value, MODULO):
    if value >= 0:
        return value % MODULO
    else:
        return value % MODULO - MODULO

# save array of difference list
def saveArray(arr, name):

    arrToSave = np.array(arr)
    arrToSave = np.reshape(arrToSave, (len(arrToSave), 1))
    arrToSave = pd.DataFrame(arrToSave)
    arrToSave.to_csv(name + '_dif.csv')

# make and save the difference list
def makeDifs(name):

    # load original dataset
    dataset = pd.read_csv(name + '.csv')['Sequence']

    # make the list of difs
    difs_list = []
    
    for i in range(len(dataset)):
        values = dataset[i].split(',')

        # use modulo to not exceed -2^64 or +2^64
        difs = []
        for j in range(len(values)-1):
            difs.append(str(modulo(int(values[j+1]) - int(values[j]), MODULO)))

        difs = ','.join(difs)
        difs_list.append(difs)

    # save the list of difs
    saveArray(difs_list, name)

if __name__ == '__main__':

    print('making dif for training data ...')
    makeDifs('train')

    print('making dif for test data ...')
    makeDifs('test')

    
