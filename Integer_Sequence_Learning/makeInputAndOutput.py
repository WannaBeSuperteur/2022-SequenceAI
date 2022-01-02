import numpy as np
import pandas as pd

# extract input and output data from array
def extractFromArray(arr, ws):

    # make numeric
    for i in range(len(arr)): arr[i] = int(arr[i])

    # extract input and output
    input_ = []
    output_ = []

    for i in range(len(arr) - ws):
        input_.append(arr[i:i+ws])
        output_.append(arr[i+ws:i+ws+1])

    # return
    return (input_, output_)

# make inputs and output data
def makeInputAndOutput(name, ws):

    # load original dataset
    difDataset = pd.read_csv(name + '_dif.csv', index_col=0)
    
    # make input and output data
    inputs, outputs = [], []
    
    for i in range(len(difDataset)):
        data = difDataset.iloc[i, 0]

        # to prevent "object of type 'float' has no len()" error
        if str(data) != 'nan':
            data = data.split(',')
            
            # extract input and output data from array
            # ignore data rows that cannot make input and output
            if len(data) >= ws + 1:
                (input_, output_) = extractFromArray(data, ws)
                
                inputs += input_
                outputs += output_

    return (inputs, outputs)

# save inputs and output data
def saveData(data, name):
    data = np.array(data)
    data = pd.DataFrame(data)
    data.to_csv(name + '.csv')

if __name__ == '__main__':

    WINDOW_SIZE = 15
    LIMIT       = 100000

    print('making input and output for training data ...')
    (train_inputs, train_outputs) = makeInputAndOutput('train', WINDOW_SIZE)

    print('making input and output for test data ...')
    (test_inputs, test_outputs) = makeInputAndOutput('test', WINDOW_SIZE)

    print('saving input and output ...')
    saveData(train_inputs[:LIMIT], 'train_input')
    saveData(train_outputs[:LIMIT], 'train_output')
    saveData(test_inputs[:LIMIT], 'test_input')
    saveData(test_outputs[:LIMIT], 'test_output')
