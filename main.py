# import matlab.engine
import pickle
import torch
import time

ADDPATH_PATH = "./addPath.m"

# start matlab engine and add path
# eng = matlab.engine.start_matlab()
# eng.run(ADDPATH_PATH, nargout=0)

def getEdges(adjMat: torch.Tensor):
    upperTriangle = torch.triu(adjMat, diagonal=1).nonzero()
    sortedIndices = upperTriangle[upperTriangle[:, 1].argsort(stable=True)]
    reversedIndices = torch.flip(sortedIndices, [1])
    return torch.cat((sortedIndices, reversedIndices), dim = 0)
    

# def convertAffinity(affinity_mat: torch.Tensor, eg1, eg2, n1, n2):  
#     # nodeAffinity = eng.zeros(n1, n2)
#     # start_time = time.time()
#     # for i in range(n1):
#     #     for j in range(n2):
#     #         nodeAffinity[i][j] = affinity_mat[i][j][i][j].item()
#     # print("Time: ", time.time() - start_time)

#     start_time = time.time()
#     # nodeAffinity = matlab.double([[affinity_mat[i][j][i][j].item() for j in range(n2)] for i in range(n1)])
#     nodeAffinity = eng.ones(n1, n2)
#     # nodeAffinity = torch.ones(n1, n2).tolist()
#     print("Time: ", time.time() - start_time)
    
#     m1 = int(eng.size(eg1, 2))
#     m2 = int(eng.size(eg2, 2))
#     # edgeAffinity = eng.zeros(m1, m2)

#     for i in range(m1):
#         for j in range(m2):
#             edgeAffinity[i][j] = affinity_mat[int(eg1[0][i])-1][int(eg2[0][j])-1][int(eg1[1][i])-1][int(eg2[1][j])-1].item()
            
#     start_time = time.time()
#     # edgeAffinity = matlab.double([[affinity_mat[int(eg1[0][i])-1][int(eg2[0][j])-1][int(eg1[1][i])-1][int(eg2[1][j])-1].item() for j in range(m2)] for i in range(m1)])
#     edgeAffinity = eng.ones(m1, m2)
#     # edgeAffinity = torch.ones(m1, m2).tolist()
#     print("Time: ", time.time() - start_time)
    
#     return nodeAffinity, edgeAffinity

# Python interface for fgm
def costomizedInterface(adjMat1: torch.Tensor, adjMat2: torch.Tensor, nodeAffinity: torch.Tensor, edgeAffinity: torch.Tensor, ct: list = None):

    # convert torch tensor to matlab double
    adjMat1, adjMat2 = matlab.double(adjMat1.tolist()), matlab.double(adjMat2.tolist())
    n1, n2 = adjMat1.size[0], adjMat2.size[0]

    # generate parameters
    par1 = eng.st("link", "cus", "val", adjMat1)
    par2 = eng.st("link", "cus", "val", adjMat2)

    # generate graph using fgm code
    g1 = eng.newGphA(pt1, par1, nargout=1)
    g2 = eng.newGphA(pt2, par2, nargout=1)

    # convert affinities to fgm format
    KP, KQ = matlab.double(nodeAffinity.tolist()), matlab.double(edgeAffinity.tolist())

    # set parameters for affinity gen
    parKnl = eng.st("alg", "cos", "KP", KP, "KQ", KQ)
    gphs = [g1, g2]

    if ct is None:
        ct = eng.ones(eng.size(KP))
    else:
        ct = matlab.double(ct.tolist())

    start_time = time.time()
    asg = eng.fgmD(KP, KQ, ct, gphs, [], [])
    print("Time: ", time.time() - start_time)
    return(list(asg['X']))

def loadData(dataFile):
    with open(dataFile, 'rb') as f:
        return pickle.load(f)
        
DIFFUTILS_DATA = "./data/diffutils.dat"

if __name__ == "__main__":
    nodeAffinity, edgeAffinity, adjMat1, adjMat2 = loadData(DIFFUTILS_DATA)
    print(nodeAffinity.shape)
    print(edgeAffinity.shape)
    print(adjMat1.shape)
    print(adjMat2.shape)
