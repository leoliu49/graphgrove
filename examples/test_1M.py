import time
import numpy as np
import pickle
from graphgrove.sgtree import NNS_L2 as SGTree_NNS_L2
from graphgrove.covertree import NNS_L2 as CoverTree_NNS_L2

gt = time.time


cores = 4

print('======== Building Dataset ==========')

d = np.load('/drive_sdc/msmarco-mpnet/mpnet-msmarco-train-embeddings-default-768.npz')
P = d['passage_embeddings'][:1000000]
Q = d['query_embeddings'][:1000]

N = P.shape[0]
D = P.shape[1]
P = np.require(P, requirements=['A','C','O','W'])

print('======== SG Tree ==========')
t = gt()
ct = SGTree_NNS_L2.from_matrix(P, use_multi_core=cores, new_base=1.3)
b_t = gt() - t
#ct.display()
print("Building time:", b_t, "seconds")

ct.dump_tree('MSMARCO_1Mx1000_base_1.3.json')

print('Test k-Nearest Neighbours - Exact (k=1): ')
t = gt()
idx1, d1 = ct.NearestNeighbour(Q, use_multi_core=cores, filename='MSMARCO_1Mx1000_base_1.3.trace')
b_t = gt() - t
print("Query time - Exact:", b_t, "seconds")


# G = pickle.load(open('/drive_sdc/msmarco-mpnet/mpnet-msmarco-train-embeddings-default-768-QP_gold.pkl','rb'))


