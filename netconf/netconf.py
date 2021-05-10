# Import packages
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

## Functions running netconf graph belief propagation
# Adapted from Matlab script found at 
# https://github.com/dhivyaeswaran/dhivyaeswaran.github.io/tree/master/code

## Import numpy arrays from csv format
def csvread(filename):
  return np.array(pd.read_csv(filename, header=None))

## Transfer edges array to adjacency matrix (N*N (typically sparse))
def edges_to_adjmat(edges):
    n = np.max(edges)+1 # Get number of nodes
    edges = np.unique(np.append(edges,edges[:,[1,0]],axis=0),axis=0) 
    edges = edges[edges[:,0]!=edges[:,1],:]
    adj_mat = csr_matrix((np.ones(len(edges[:,0])),(edges[:,0],edges[:,1])),shape=(n,n)).toarray()
    return adj_mat

  
## Run netConf belief propagation
## Input variables
# edges - csv of 
# priors - N*k priors (set to 1/k..1/k for unseeded nodes)
# mod - k*k modulation matrix
# ep - 0.5 - ep/rho(A) is the modulation multiplier
# stop - 0 - Defines fixed number of iterations
# verbose - False - output loss for each iteration
# max_iter - 100 - maximum iterations
# limit - 1e-4 - loss threshold for early stop 
def netconf(edges,priors,mod=np.eye(2),
            ep=0.5,stop=0,verbose=False,max_iter=100,limit=1e-4):  
  # Define initial variables
  if verbose: print('Nodes: {}, Edges: {}'.format(len(priors),len(edges)))
  t = time.time() # Begin timing
  B, [N,k] = priors, priors.shape
  l, diff1 = np.zeros(N), 1
  adj_mat = edges_to_adjmat(edges)
  v, D = np.abs(eigsh(adj_mat,1)[0][0]), np.diag(sum(adj_mat))
  M = np.dot(np.divide(ep,v),mod)
  M1 = np.dot(M,np.linalg.pinv(np.eye(k)-np.dot(M,M)))
  M2 = np.dot(M,M1)
  
  # Add headers for verbose columns
  if verbose: print('It\tLoss\tLabel change\n')
  
  # Define number of iterations
  n_iter = max_iter if not stop else stop
  
  # Loop through iterations
  for i in range(n_iter-1): 
    # Break when loss below limit when iterations not specified
    if not stop and diff1 < limit:
      if not verbose: print(i,' iterations')
      break
    
    # Update beliefs
    Bold, lold = B, l
    B = priors + np.dot(np.dot(adj_mat,B),M1) - np.dot(np.dot(D,B),M2)
    l = B.argmax(1)
    diff2 = np.sum(lold!=l)
    diff1 = np.max(np.abs(B-Bold))
    if verbose: print('{}\t{:.5e}\t\t{}\n'.format(i,diff1,diff2))
  
  # Return final beliefs and elapsed time
  if verbose: print('Time elapsed: {} seconds'.format(time.time()-t))
  return B, time