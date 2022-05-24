import pickle
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


def generate_seq(data, train_length, pred_length):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: 1]
    return np.split(seq, 2, axis=1)

def generate_from_train_val_test(data, transformer):
    mean = None
    std = None
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(data[key], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        # yield (x - mean) / std, y
        yield x, y

def generate_from_data(data, length, transformer):
    mean = None
    std = None
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for line1, line2 in ((0, train_line),
                         (train_line, val_line),
                         (val_line, length)):
        x, y = generate_seq(data['data'][line1: line2], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        # yield (x - mean) / std, y
        yield x, y


def generate_data(graph_signal_matrix_filename, transformer=None):
    '''
    shape is (num_of_samples, 12, num_of_vertices, 1)
    '''
    data = np.load(graph_signal_matrix_filename)
    # print(data['data'].shape) #(16992, 307, 3)
    # sys.exit(0)
    keys = data.keys()
    if 'train' in keys and 'val' in keys and 'test' in keys:
        for i in generate_from_train_val_test(data, transformer):
            yield i
    elif 'data' in keys:
        length = data['data'].shape[0]
        for i in generate_from_data(data, length, transformer):
            yield i
    else:
        raise KeyError("neither data nor train, val, test is in the data")



class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()



def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


import json

def load_dataset(config_filename, batch_size, valid_batch_size= None, test_batch_size=None):
    with open(config_filename, 'r') as f:
        config = json.loads(f.read())

    num_of_vertices = config['num_of_vertices']
    adj_filename = config['adj_filename']
    id_filename = config['id_filename']
    if id_filename is not None:
        if not os.path.exists(id_filename):
            id_filename = None

    adj = get_adjacency_matrix(adj_filename, num_of_vertices,
                               id_filename=id_filename)

    adj_dtw = np.array(pd.read_csv(config['adj_dtw_filename'], header=None))

      
    graph_signal_matrix_filename = config['graph_signal_matrix_filename']
        
    
    data = {}
    for idx, (x, y) in enumerate(generate_data(graph_signal_matrix_filename)):
        category = ['train', 'val', 'test'][idx]
        data['x_' + category] = x[..., :1].astype(np.float32) 
        data['y_' + category] = y[..., :1].astype(np.float32)
        print(x.shape, y.shape)
        # (10172, 12, 307, 1) (10172, 12, 307)
        # (3375, 12, 307, 1) (3375, 12, 307)
        # (3376, 12, 307, 1) (3376, 12, 307)
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    import copy
    # data['y_train_cl'] = copy.deepcopy(data['y_train'])[:, :, 0:1, 1:3] #bt12
    data['y_train_cl'] = copy.deepcopy(data['y_train'])[:, :, 0:1, -2:] #bt12
    # data['y_train_cl'][..., 0] = scaler.transform(data['y_train'][..., 0])

    # data['train_loader'] = DataLoaderM_new(data['x_train'], data['y_train'], data['y_train_cl'], batch_size)
    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
#     data['scaler'] = StandardScaler(mean=0, std=1) #不做inverse transform时使用
    return data, adj, adj_dtw, config, num_of_vertices


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    z = torch.tensor((x-mean)/std,dtype=torch.float)
    return z


def load_adj(pkl_filename, adjtype = "doubletransition"):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_adj_dtw(pkl_filename):
    adj_dtw = np.array(pd.read_csv(pkl_filename, header=None))
    return asym_adj(adj_dtw)


###########################################################################################


def construct_tg1(A, A_dtw):
    '''
    [t, 0, 
     0, t]
    '''

    N = len(A)
    adj = np.zeros([N * 2] * 2) # "steps" = 4 !!!
    
    adj[0: N, 0: N] = A_dtw
    adj[N: 2*N, N: 2*N] = A_dtw

    return adj

def construct_tg2(A, A_dtw):
    '''
    [0, T, 
     T, 0]
    '''

    N = len(A)
    adj = np.zeros([N * 2] * 2) # "steps" = 4 !!!
    
    adj[0: N, N: 2*N] = A_dtw
    adj[N: 2*N, 0: N] = A_dtw

    return adj

def construct_sg1(A, A_dtw):
    '''
    [s, 0, 
     0, s]
    '''

    N = len(A)
    adj = np.zeros([N * 2] * 2) # "steps" = 4 !!!
    
    adj[0: N, 0: N] = A
    adj[N: 2*N, N: 2*N] = A

    return adj

def construct_sg2(A, A_dtw):
    '''
    [0, s, 
     s, 0]
    '''

    N = len(A)
    adj = np.zeros([N * 2] * 2) # "steps" = 4 !!!
    
    adj[0: N, N: 2*N] = A
    adj[N: 2*N, 0: N] = A

    return adj

def construct_tc(A, A_dtw):
    '''
    [0, i, 
     i, 0]
    '''

    N = len(A)
    adj = np.zeros([N * 2] * 2) # "steps" = 4 !!!
    tc = np.eye(N, N)
    
    adj[0: N, N: 2*N] = tc
    adj[N: 2*N, 0: N] = tc

    return adj

def construct_tg_sg1(A, A_dtw):
    '''
    [t, 0, 
     0, s]
    '''

    N = len(A)
    adj = np.zeros([N * 2] * 2) # "steps" = 4 !!!
    
    adj[0: N, 0: N] = A_dtw
    adj[N: 2*N, N: 2*N] = A

    return adj

def construct_tg_sg2(A, A_dtw):
    '''
    [s, 0, 
     0, t]
    '''

    N = len(A)
    adj = np.zeros([N * 2] * 2) # "steps" = 4 !!!
    
    adj[0: N, 0: N] = A
    adj[N: 2*N, N: 2*N] = A_dtw

    return adj
