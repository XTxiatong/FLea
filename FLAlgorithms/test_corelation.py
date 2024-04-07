import numpy as np
import torch

def pairwise_dist(A):
    # Taken frmo https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    #A = torch_print(A, [torch.reduce_sum(A)], message="A is")
    r = torch.sum(A*A, 1)
    r = torch.reshape(r, [-1, 1])
    rr = r.repeat(1,A.shape[0])
    rt = r.T.repeat(A.shape[0],1)
    D = torch.maximum(rr - 2*torch.matmul(A, A.T) + rt, 1e-7*torch.ones(A.shape[0], A.shape[0]))
    D = torch.sqrt(D)
    return D

def dist_corr(X, F):
    n = X.shape[0]
    a = pairwise_dist(X)
    b = pairwise_dist(F)

    A = a - torch.mean(a,1).repeat(a.shape[1],1).T - torch.mean(a,0).repeat(a.shape[0],1) + torch.mean(a)
    B = b - torch.mean(b,1).repeat(b.shape[1],1).T - torch.mean(b,0).repeat(b.shape[0],1) + torch.mean(b)
    dCovXY = torch.sqrt(torch.sum(A*B) / (n ** 2)+ 1e-7)
    dVarXX = torch.sqrt(torch.sum(A*A) / (n ** 2)+ 1e-7)
    dVarYY = torch.sqrt(torch.sum(B*B) / (n ** 2)+ 1e-7)
    dCorXY = dCovXY / (torch.sqrt(dVarXX + 1e-7) * torch.sqrt(dVarYY+ 1e-7) )
    return dCorXY
    
A = torch.tensor([[1,2,3],[2,2,3]])
print(pairwise_dist(A))