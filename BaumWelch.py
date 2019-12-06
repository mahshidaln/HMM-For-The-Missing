import numpy as np


def logSumExp(a):
    if np.all(np.isinf(a)):
        return np.log(0)
    else:
        b = np.max(a)
        return(b + np.log(np.sum(np.exp(a-b))))


def pForward(g, x):
    pXf = logSumExp(g[len(x)-1,:])
    return(pXf)


def forward(notes_no, hidden_no, un, initial, transition, emission, notes_index):
    g = np.zeros((notes_no,hidden_no))
   
    for i in range(0,hidden_no):
        g[0,i] = (initial[i]) + (emission[i, notes_index[0]])
    
    
    for j in range(1, notes_no):
        for l in range(0, hidden_no):
            g[j,l] = logSumExp(np.asarray(g[j-1, :])+np.asarray(transition[:,l])+(emission[l,notes_index[j]]))
    return(g)


def backward(notes_no, hidden_no, un, initial, transition, emission, notes_index):
    r = np.zeros((notes_no,hidden_no))
    for j in range(notes_no-2, -1, -1):
        for l in range(0, hidden_no):
            r[j, l] = logSumExp(np.asarray(r[j+1,: ]) + np.asarray(transition[l,:]) + emission[:, notes_index[j+1]])
    
    return(r)


def baum_welch(notes_no, hidden_no, un, notes_index, threshold):
    vals = np.random.rand(hidden_no)
    initial = np.log(vals/np.sum(vals))
    Tmat = np.zeros(shape = (hidden_no, hidden_no))
    emission = np.zeros(shape = (hidden_no, un))
    gamma = np.zeros(shape = (notes_no, hidden_no))
    beta = np.zeros(shape = (notes_no,hidden_no,hidden_no))
    iterations = 0 
    convergence = 0 
    count = 0
    pOld = 1E10
    pNew = 0
    criteria = 0
    #cdef double[:,:] p = np.zeros(shape = (n,m))
    
    vals1 = np.random.rand(hidden_no,hidden_no)
    vals2 = np.random.rand(hidden_no,un)
    Tmat = np.log(vals1/np.sum(vals1, axis=1)[:,None])
    emission = np.log(vals2/np.sum(vals2, axis = 1)[:,None])
    
    while convergence == 0:
        g = forward(notes_no, hidden_no, un, initial, Tmat, emission, notes_index)
        h = backward(notes_no, hidden_no, un, initial, Tmat, emission, notes_index)
        pNew = pForward(g, notes_index)
        
        for t in range(0, notes_no):
            for i in range(0,hidden_no):
                gamma[t,i] = g[t,i] + h[t,i] - pNew
        #p = np.full((n,m), pNew)
        #gamma = g+h-p
        for t in range(1, notes_no):
            for i in range(0, hidden_no):
                for j in range(0, hidden_no):
                    beta[t,i,j] = Tmat[i,j] + emission[j, notes_index[t]] + g[t-1, i] + h[t, j] - pNew
    
    
        initial = gamma[0,:] - logSumExp(gamma[0,:])
        for i in range(0, hidden_no):
            for j in range(0, hidden_no):
                Tmat[i,j] = logSumExp(beta[1::, i, j]) - logSumExp(beta[1::, i,:])
        for i in range(0,hidden_no):
            for w in range(0, un):
                j = 0
                count = 0
                for t in range(0,notes_no):
                    if notes_index[t] == w:
                        count = count+1
                indicies = np.zeros(count)
                for t in range(0,notes_no):
                    if notes_index[t] == w:
                        indicies[j] = gamma[t,i]
                        j = j+1
                    
                emission[i,w] = logSumExp(indicies) - logSumExp(gamma[:,i])
        
        criteria = abs(pOld - pNew)
        if criteria < threshold:
            convergence = 1
        
        elif iterations > 1000:
            convergence = 1
        else:
            convergence = 0
            pOld = pNew
            iterations +=1
            print(iterations)
    return (iterations, pNew, np.exp(initial), np.exp(emission), np.exp(Tmat))