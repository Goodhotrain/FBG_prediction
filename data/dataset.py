import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class FBGDataset(Dataset):
    def __init__(self, args, subset):
    
        self.ni_data = pd.read_csv(args.nutrition_path)
        self.label_data = pd.read_csv(args.label_path)
        self.static_data = pd.read_csv(args.static_path)
        self.m = []
        self.label = []
        for i,(l, x) in enumerate( self.label_data.values):
            if subset == None:
                self.m.append(i)
                self.label.append(l)            
            else:
                if x == subset:
                    self.m.append(i)
                    self.label.append(l)
        self.standard = StandardScaler()
        self._preprocess_static_data()
        self._normalize_per_column()


    def _preprocess_static_data(self):
        static_features = self.static_data.drop(columns=['UserID','Day']).values
        static_scaled = self.standard.fit_transform(static_features)
        self.staic_processed = pd.DataFrame(static_scaled, columns=self.static_data.columns[2:])
        # self.pe_data_processed['UserID'] = self.static_data['UserID'].values


    def _normalize_per_column(self):
        ni_features = self.ni_data.drop(columns=['UserID', 'Day']).values
        ni_user_ids = self.ni_data['UserID']
        ni_days = self.ni_data['Day']

        ni_features = np.where(ni_features > 0, ni_features, 1e-10)  # replace 0 with a very small value
        ni_features_log = np.log(ni_features)

        scaler = StandardScaler()
        ni_scaled = scaler.fit_transform(ni_features_log)

        if np.any(np.isnan(ni_scaled)):
            print("NaN found during normalization, replacing with 0")
            ni_normalized = np.nan_to_num(ni_scaled, nan=0.0)

        self.ni_data_processed = pd.DataFrame(ni_scaled, columns=self.ni_data.columns[2:])
        self.ni_data_processed['UserID'] = ni_user_ids
        self.ni_data_processed['Day'] = ni_days

    def __len__(self):
        return len(self.m)

    def __getitem__(self, idx):
        id = self.m[idx]
        static_dim = 32
        static_data = self.staic_processed.iloc[id].values
        # print(self.ni_data_processed.index)
        UserID = self.ni_data_processed.iat[id, self.ni_data_processed.columns.get_loc('UserID')]
        day = self.ni_data_processed.iat[id, self.ni_data_processed.columns.get_loc('Day')]
        
        # ni
        ni = self.ni_data_processed[(self.ni_data_processed['UserID'] == UserID) & (self.ni_data_processed['Day'] <= day)]
        # history FBG
        ni = ni.drop(columns=['UserID','Day']).values
        max_time_steps = 16
        if ni.shape[0] < max_time_steps:
            # print(ni.shape)
            ni = np.pad(ni, ((0, max_time_steps - ni.shape[0]), (0, 0)), mode='constant')
        else:
            print(ni.shape)
            ni = ni[-max_time_steps:] if ni.shape[0] > max_time_steps else ni
        # label
        label = self.label[idx]
        return id, torch.tensor(static_data, dtype=torch.float32), \
               torch.tensor(ni, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

