from torch.utils.data import Dataset
import glob as glob
import numpy as np
import pandas as pd


# instantiate dataset class using preexisiting torch dataset
class POEMMS_Dataset(Dataset):
    def __init__(self, file_dir, transform=None, target_transform=None):
        self.file_dir = file_dir                            # read in path to file directory
        self.files = glob.glob(self.file_dir)               # read in all csv files in the directory
        
        self.solutions = []                                 # create solutions list
        for file in self.files:                             # loop through all files
            for s in file.split('/')[-2].split('_'):        # get solutions based off of file name
                if s != '' and s not in self.solutions:     # if solution is not already in the list add it
                    self.solutions.append(s)
        self.one_hot_labels = {}                            # create one-hot label dictionary
        for file in self.files:                             # loop through all files
            z = np.zeros(len(self.solutions))               # create a list of all zeros
            file_label = file.split('/')[-2].split('_')     # get solutions based off of file name
            for f in file_label:                            # loop through all solutions in the file
                if f != '':                                 # ensure a real solution
                    z[self.solutions.index(f)] = 1          # set index of solution in the list of zeros equal to 1
            self.one_hot_labels[file] = z                   # set the file name with the one-hot labeled list in the dictionary
    

    def __len__(self):                                      # create len function
        return len(self.files)                              # return len of the dataset
    
    def __getitem__(self, idx):                             # create get item function
        file = self.files[idx]      
        #print(file)
        # select a file from list of files based off of index
        data = pd.read_pickle(file)

        #data = pd.read_csv(file, index_col=0)               # read in data from the file
        label = self.one_hot_labels[file]                   # fetch file label from one-hot labels dictionary
        data = data.to_numpy()                              # convert data to numpy array
        
        return np.array([data]), label                      # return formatted data and corresponding label
