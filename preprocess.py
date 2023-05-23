
import pickle
import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, in_data_size=195, n_bins=15):
        
        self.in_size = in_data_size
        self.n_bins = n_bins
        self.nunique = np.zeros(in_data_size, dtype=int)
        self.ohe_edges = np.zeros(in_data_size, dtype=int)
        self.ohe_check = np.zeros(in_data_size, dtype=int)
        self.ohe_vals = np.zeros((in_data_size, n_bins))
        self.out_data_size = 0

    def analyse_data(self, data):
        start_index = 0
        for i, col in enumerate(data.columns):
            self.nunique[i] = data[col].nunique()
            if self.nunique[i] > self.n_bins or self.nunique[i] <= 2:
                self.ohe_edges[i] = start_index
                start_index+=1
            else:
                self.ohe_check[i] = 1
                self.ohe_edges[i] = start_index
                unique_vals = data[col].unique().astype(np.float64)
                unique_vals = np.sort(unique_vals)
                for j, val in enumerate(unique_vals):
                    self.ohe_vals[i][j] = round(val*100)
                start_index += self.nunique[i]+1
        self.out_data_size = int(start_index)

    def one_hot_encode(self, data_arr):
        ohe_arr = np.zeros(self.out_data_size, dtype=np.float32)
        for i, value in enumerate(data_arr):
            # Check if the index is in self.ohe_vals
            if self.ohe_check[i] == 1:
                testval = round(value*100)
                for j in range(int(self.nunique[i])):
                    if (testval == self.ohe_vals[i][j]):
                        ohe_arr[int(self.ohe_edges[i] + j)] = 1
                        break
            else:
                # If it's not, directly copy the value
                ohe_arr[int(self.ohe_edges[i])] = value
            
        return ohe_arr

    
    def unhot(self, ohe_arr):
        # Initialize the output array
        data_arr = np.zeros(self.in_size, dtype=np.float32)
        # Iterate over ohe_arr
        for i in range(self.in_size):
            # Check if the index was one-hot encoded
            if self.ohe_check[i] == 1:
                ohe_index = np.where(ohe_arr[int(self.ohe_edges[i]):int(self.ohe_edges[i] + self.nunique[i] + 1)] == 1)
                data_arr[i] = self.ohe_vals[i][ohe_index]/100.
            else:
                data_arr[i] = ohe_arr[int(self.ohe_edges[i])]
        return data_arr
    

    def save_state(self, filename):
        state = {
            'in_data_size' : self.in_size,
            'n_bins' : self.n_bins,
            'out_size' : self.out_data_size,
            'nunique': self.nunique,
            'ohe_edges': self.ohe_edges,
            'ohe_check': self.ohe_check,
            'ohe_vals': self.ohe_vals,
            'out_data_size': self.out_data_size
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        self.n_bins = state['n_bins']
        self.in_size = state['in_data_size']
        self.out_data_size = state['out_size']
        self.nunique = state['nunique']
        self.ohe_edges = state['ohe_edges']
        self.ohe_check = state['ohe_check']
        self.ohe_vals = state['ohe_vals']
        self.out_data_size = state['out_data_size']

    def get_check_edge_nuni(self):
        return self.ohe_check, self.ohe_edges, self.nunique, self.in_size

   

## saving usage example


data = pd.read_csv("operator-presets2-norm_mod.csv")
data = data.iloc[:, :-1]
data.fillna(0., inplace=True)

preproc = Preprocessor(in_data_size=195, n_bins=15)

preproc.analyse_data(data)
preproc.save_state("preproc_state.pkl")


## loading usage example

'''
data = pd.read_csv("operator-presets2-norm_mod.csv")
data = data.iloc[:, :-1]
data.fillna(0., inplace=True)
data_arr = data.iloc[[1]].to_numpy()[0]

preproc = Preprocessor(in_data_size=195, n_bins=15)
preproc.load_state("preproc_state.pkl")
ohe_arr = preproc.one_hot_encode(data_arr)
print(ohe_arr.size)
output_arr = preproc.unhot(ohe_arr)
'''

# usage example
# that also checks how long it takes

'''
import time

data = pd.read_csv("operator-presets2-norm_mod.csv")
data = data.iloc[:, :-1]
data.fillna(0., inplace=True)

preproc = Preprocessor()

preproc.analyse_data(data)
data_arr = data.iloc[[1]].to_numpy()[0]


start_time = time.perf_counter() ##### START TIMER
onehot_arr = preproc.one_hot_encode(data_arr)
print(f"one hot: {(time.perf_counter() - start_time) * 1000} ms.")

start_time = time.perf_counter()
unhot_arr = preproc.unhot(onehot_arr)
print(f"unhot: {(time.perf_counter() - start_time) * 1000} ms.")

# check if needed
new_df = pd.DataFrame([data_arr, unhot_arr], columns=data.columns) 
new_df.to_csv('check_preproc2.csv', index=False)
'''
