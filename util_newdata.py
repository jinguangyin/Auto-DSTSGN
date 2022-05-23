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

class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2):
        self.P = window
        self.h = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

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

class DataLoaderM_new(object):
    def __init__(self, xs, ys, ycl, batch_size, pad_with_last_sample=True):
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
            ycl_padding = np.repeat(ycl[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            ycl = np.concatenate([ycl, ycl_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.ycl = ycl

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, ycl = self.xs[permutation], self.ys[permutation], self.ycl[permutation]
        self.xs = xs
        self.ys = ys
        self.ycl = ycl


    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                y_i_cl = self.ycl[start_ind: end_ind, ...]
                yield (x_i, y_i, y_i_cl)
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

# def load_adj(pkl_filename):
#     sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
#     return adj


# def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
#     data = {}
#     for category in ['train', 'val', 'test']:
#         cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
#         data['x_' + category] = cat_data['x'].astype(np.float32)
#         data['y_' + category] = cat_data['y'].astype(np.float32)
#     scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
#     # Data format
#     for category in ['train', 'val', 'test']:
#         data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

#     import copy
#     # data['y_train_cl'] = copy.deepcopy(data['y_train'])[:, :, 0:1, 1:3] #bt12
#     data['y_train_cl'] = copy.deepcopy(data['y_train'])[:, :, 0:1, -2:] #bt12
#     # data['y_train_cl'][..., 0] = scaler.transform(data['y_train'][..., 0])


#     data['train_loader'] = DataLoaderM_new(data['x_train'], data['y_train'], data['y_train_cl'], batch_size)
#     data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
#     data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
#     data['scaler'] = scaler
#     return data



from utils_4n0_3layer_12T_res import (generate_data,get_adjacency_matrix,
                       masked_mae_np, masked_mape_np, masked_mse_np)
import json
def load_dataset(config_filename, batch_size, valid_batch_size= None, test_batch_size=None, if_causal = False):
    with open(config_filename, 'r') as f:
        config = json.loads(f.read())
    
    # module_type = config['module_type']
    # act_type = config['act_type']
    # temporal_emb = config['temporal_emb']
    # spatial_emb = config['spatial_emb']
    # use_mask = config['use_mask']
    # num_of_features = config['num_of_features']
    # points_per_hour = config['points_per_hour']
    # num_for_predict = config['num_for_predict']
    # batch_size = config['batch_size']

    num_of_vertices = config['num_of_vertices']
    adj_filename = config['adj_filename']
    id_filename = config['id_filename']
    if id_filename is not None:
        if not os.path.exists(id_filename):
            id_filename = None

    adj = get_adjacency_matrix(adj_filename, num_of_vertices,
                               id_filename=id_filename)
    #adj_mx = construct_adj(adj, 3)
    adj_dtw = np.array(pd.read_csv(config['adj_dtw_filename'], header=None))
    #xxx
    adj_mx = construct_adj_fusion(adj, adj_dtw, 4, if_causal)
    print("The shape of localized adjacency matrix: {}".format(
        adj_mx.shape), flush=True)
      
    graph_signal_matrix_filename = config['graph_signal_matrix_filename']
        
    
    data = {}
    for idx, (x, y) in enumerate(generate_data(graph_signal_matrix_filename)):
        category = ['train', 'val', 'test'][idx]
#     for category in ['train', 'val', 'test']:
#         cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = x[..., :1].astype(np.float32) #未保留输入的另外几维
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


    data['train_loader'] = DataLoaderM_new(data['x_train'], data['y_train'], data['y_train_cl'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
#     data['scaler'] = StandardScaler(mean=0, std=1) #不做inverse transform时使用
    return data, adj_mx, config, num_of_vertices


# def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
#     data = {}
#     for category in ['train', 'val', 'test']:
#         cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
#         data['x_' + category] = cat_data['x']
#         data['y_' + category] = cat_data['y']
#     scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
#     # Data format
#     for category in ['train', 'val', 'test']:
#         data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
#
#     data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size, pad_with_last_sample=False)
#     data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size, pad_with_last_sample=False)
#     data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size, pad_with_last_sample=False)
#     data['scaler'] = scaler
#     return data


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


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))



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





def construct_adj(A, steps):


    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # print(np.sum(A))
    # print(np.sum(A, axis=0))
    # print(np.sum(A, axis=1))
    # print(np.where(A!=np.transpose(A)))
    # print(A)

    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # sys.exit(0)

    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    '''
    N = len(A)
    adj = np.zeros([N * steps] * 2)

    for i in range(steps):
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


def construct_adj_fusion(A, A_dtw, steps, if_causal = False):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)

    ----------
    This is 4N_1 mode:

    [T, 1, 1, T
     1, S, 1, 1
     1, 1, S, 1
     T, 1, 1, T]

    '''

    N = len(A)
    adj = np.zeros([N * steps] * 2) # "steps" = 4 !!!
    
    #'''李府显：次对角线6个单位矩阵
    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1
    #'''李府显：次次次对角线2个
    adj[3 * N: 4 * N, 0:  N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[0 : N, 3 * N: 4 * N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]
    #李府显：次次对角线4个单位矩阵（）其实等号右侧放次对角线上哪个都行
    adj[2 * N: 3 * N, 0 : N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[0 : N, 2 * N: 3 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]


    # for i in range(len(adj)): #李府显：对于我们的数据集，有没有这个操作都行
    #     adj[i, i] = 1

    if if_causal:
        adj = np.tril(adj, k=0) # https://www.thinbug.com/q/8905501


    #李府显：对角线4个
    for i in range(steps):
        if (i == 1) or (i == 2):
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A # A_dtw
        else:
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw

    print(adj)
    return adj


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
