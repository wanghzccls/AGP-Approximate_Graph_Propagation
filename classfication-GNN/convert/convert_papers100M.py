import argparse
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj
from ogb.nodeproppred import PygNodePropPredDataset
import sklearn.preprocessing
import numpy as np
import tracemalloc
import gc
import struct
    
def papers100M():
    dataset=PygNodePropPredDataset("ogbn-papers100M")
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    feat=data.x.numpy()
    feat=np.array(feat,dtype=np.float64)

    #normalize feats
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)

    #save feats
    np.save('../data/papers100M_feat.npy',feat)
    del feat
    gc.collect()

    print('making the graph undirected')
    data.edge_index=to_undirected(data.edge_index,data.num_nodes)
    row,col=data.edge_index

    N=data.num_nodes
    row=row.numpy()
    col=col.numpy()
    adj=sp.csr_matrix((np.ones(row.shape[0]),(row,col)),shape=(N,N))
    adj=adj+sp.eye(adj.shape[0])
    EL=adj.indices
    PL=adj.indptr

    del adj
    gc.collect()

    EL=np.array(EL,dtype=np.uint32)
    PL=np.array(PL,dtype=np.uint32)
    EL_re=[]

    for i in range(1,PL.shape[0]):
        EL_re+=sorted(EL[PL[i-1]:PL[i]],key=lambda x:PL[x+1]-PL[x])
    EL_re=np.asarray(EL_re,dtype=np.uint32)

    #"save graph
    f1=open('../data/papers100M_adj_el.txt','wb')
    for i in EL_re:
        m=struct.pack('I',i)
        f1.write(m)
    f1.close()

    f2=open('../data/papers100M_adj_pl.txt','wb')
    for i in PL:
        m=struct.pack('I',i)
        f2.write(m)
    f2.close()
    del EL
    del PL
    del EL_re
    gc.collect()

    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    #get labels
    labels=data.y
    train_labels=labels.data[train_idx]
    val_labels=labels.data[val_idx]
    test_labels=labels.data[test_idx]

    train_idx=train_idx.numpy()
    val_idx=val_idx.numpy()
    test_idx=test_idx.numpy()
    train_idx=np.array(train_idx, dtype=np.int32)
    val_idx=np.array(val_idx,dtype=np.int32)
    test_idx=np.array(test_idx,dtype=np.int32)

    train_labels=train_labels.numpy().T
    val_labels=val_labels.numpy().T
    test_labels=test_labels.numpy().T

    train_labels=np.array(train_labels,dtype=np.int32)
    val_labels=np.array(val_labels,dtype=np.int32)
    test_labels=np.array(test_labels,dtype=np.int32)
    train_labels=train_labels.reshape(train_labels.shape[1])
    val_labels=val_labels.reshape(val_labels.shape[1])
    test_labels=test_labels.reshape(test_labels.shape[1])
    np.savez('../data/papers100M_labels.npz',train_idx=train_idx,val_idx=val_idx,test_idx=test_idx,train_labels=train_labels,val_labels=val_labels,test_labels=test_labels)

if __name__ == "__main__":
    papers100M()