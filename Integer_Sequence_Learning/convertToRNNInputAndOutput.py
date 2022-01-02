import pandas as pd
import numpy as np

# convert a column of dataset
def convert(df, colName):
    
    for i in range(2, 11):
        print('colName=' + str(colName) + ', converting for i=' + str(i) + '...')
        
        modVal  = df[colName].mod(i)
        modRate = modVal / (i - 1.0)  # [0 ..1]
        modRate = 2.0 * modRate - 1.0 # [-1..1]
        
        df[colName + '_div' + str(i)] = modRate

    df = df.drop(colName, 1)

    return df

# modify column order:
# [00, 01, ..., 0(x-1), 10, ..., (y-1)(x-1)] to [00, 10, ..., (y-1)0, 01, ..., (y-1)(x-1)]
def modifyColOrder(df, x, y):

    cols = df.columns.tolist()
    new_cols = []

    for i in range(x):
        for j in range(y):
            new_cols.append(cols[j * x + i])
    
    new_df = df[new_cols]
    return new_df

# save dataset
def saveDataset(dataset, name, cols):

    dataset = dataset.round(3) # to reduce the volume of *.csv files

    # save dataset (for deep learning)
    dataset.to_csv(name + '_converted.csv')

    # modify column order of the dataset
    visualDataset = modifyColOrder(dataset, 9, cols)

    # save dataset (for visualization)
    visualDataset.to_csv(name + '_converted_visual.csv')

# load and convert dataset
def loadAndConvert(name):

    # load dataset
    dataset = pd.read_csv(name + '.csv', index_col=0)
    dataset = dataset.astype('int64')
    cols = len(dataset.columns)

    # modify column names
    dataset.columns = ['col_' + str(i) for i in range(cols)]
    
    # convert
    for i in range(cols):
        dataset = convert(dataset, 'col_' + str(i))

    # save dataset
    saveDataset(dataset, name, cols)

if __name__ == '__main__':

    loadAndConvert('train_input')
    loadAndConvert('train_output')
    loadAndConvert('test_input')
    loadAndConvert('test_output')
    
    
