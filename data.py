import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn import preprocessing
import numpy as np 
import pandas as pd
import os

features = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', \
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', \
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', \
            'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', \
            'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', \
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', \
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', \
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', \
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', \
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'type']

df = pd.read_csv("data/kddcup.data_10_percent.gz", header=None, names=features)

# type mapping
df['type'] = df['type'].map(lambda x: 1 if x == "normal." else 0)
for col in ['protocol_type', 'service', 'flag']:
    df[col] = preprocessing.LabelEncoder().fit_transform(df[col])

labels = df['type']

# drop highly correlated features
data = df.drop(['num_root', 'srv_serror_rate', 'srv_rerror_rate', 'dst_host_srv_serror_rate', \
                'dst_host_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', \
                'dst_host_same_srv_rate', 'type'], axis=1)

# normalization
# data = (data - data.min()) / (data.max() - data.min())

data.to_csv('data/data.csv', header=False, index=False)
labels.to_csv('data/labels.csv', header=False, index=False)

def get_loader(hyp):
    # set random seed
    gen = torch.Generator()
    gen.manual_seed(12)

    inputs = pd.read_csv('data/data.csv')
    targets = pd.read_csv('data/labels.csv')

    dataset = TensorDataset(torch.tensor(inputs.values, dtype=torch.float32), 
                      torch.tensor(targets.values, dtype=torch.float32))

    trainset, testset = random_split(dataset, [0.8, 0.2], generator=gen)

    trainloader = DataLoader(dataset=trainset, batch_size=hyp['batch_size'], shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=hyp['batch_size']*10, shuffle=False)
    
    return trainloader, testloader
