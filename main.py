import matlab.engine
import torch
import time

ADDPATH_PATH = "./addPath.m"

# start matlab engine and add path
eng = matlab.engine.start_matlab()
eng.run(ADDPATH_PATH, nargout=0)

def convertAffinity(affinity_mat: torch.Tensor, eg1, eg2, n1, n2):  
    # nodeAffinity = eng.zeros(n1, n2)
    # start_time = time.time()
    # for i in range(n1):
    #     for j in range(n2):
    #         nodeAffinity[i][j] = affinity_mat[i][j][i][j].item()
    # print("Time: ", time.time() - start_time)

    start_time = time.time()
    # nodeAffinity = matlab.double([[affinity_mat[i][j][i][j].item() for j in range(n2)] for i in range(n1)])
    nodeAffinity = eng.ones(n1, n2)
    # nodeAffinity = torch.ones(n1, n2).tolist()
    print("Time: ", time.time() - start_time)
    
    m1 = int(eng.size(eg1, 2))
    m2 = int(eng.size(eg2, 2))
    # edgeAffinity = eng.zeros(m1, m2)

    for i in range(m1):
        for j in range(m2):
            edgeAffinity[i][j] = affinity_mat[int(eg1[0][i])-1][int(eg2[0][j])-1][int(eg1[1][i])-1][int(eg2[1][j])-1].item()
            
    start_time = time.time()
    # edgeAffinity = matlab.double([[affinity_mat[int(eg1[0][i])-1][int(eg2[0][j])-1][int(eg1[1][i])-1][int(eg2[1][j])-1].item() for j in range(m2)] for i in range(m1)])
    edgeAffinity = eng.ones(m1, m2)
    # edgeAffinity = torch.ones(m1, m2).tolist()
    print("Time: ", time.time() - start_time)
    
    return nodeAffinity, edgeAffinity

# Python interface for fgm
def costomizedInterface(adjMat1: torch.Tensor, adjMat2: torch.Tensor, affinity_mat: torch.Tensor, ct: list):

    # convert torch tensor to matlab double
    adjMat1 = matlab.double(adjMat1.tolist())
    adjMat2 = matlab.double(adjMat2.tolist())

    n1 = adjMat1.size[0]
    n2 = adjMat2.size[0]

    # generate empty node features
    pt1 = matlab.double([[0 for _ in range(n1)] for _ in range(2)])
    pt2 = matlab.double([[0 for _ in range(n2)] for _ in range(2)])

    # generate parameters
    par1 = eng.st("link", "cus", "val", adjMat1)
    par2 = eng.st("link", "cus", "val", adjMat2)

    # generate graph using fgm code
    g1 = eng.newGphA(pt1, par1, nargout=1)
    g2 = eng.newGphA(pt2, par2, nargout=1)

    # convert affinities to fgm format
    KP, KQ = convertAffinity(affinity_mat, g1["Eg"], g2["Eg"], n1, n2)

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

if __name__ == "__main__":
    NODES_NUM = 100
    M_gt = torch.zeros((NODES_NUM, NODES_NUM))
    M_gt[torch.arange(0, NODES_NUM, dtype=torch.int64), torch.randperm(NODES_NUM)] = 1

    G = torch.rand(NODES_NUM, NODES_NUM)
    G = (G + G.t() > 1.2).float()
    # G = torch.tensor([
    #     [0, 1, 0, 0, 0],
    #     [1, 0, 1, 0, 0],
    #     [0, 1, 0, 1, 1],
    #     [0, 0, 1, 0, 0],
    #     [0, 0, 1, 0, 0]
    # ]).float()

    torch.diagonal(G).fill_(0)
    g = torch.mm(torch.mm(M_gt.t(), G), M_gt)

    edges1 = torch.triu(G).nonzero()
    edges2 = torch.triu(g).nonzero()

    print(edges1.size())

    affinity = torch.sparse_coo_tensor((NODES_NUM, NODES_NUM, NODES_NUM, NODES_NUM), dtype=torch.float32)
    index = []
    for i in range(NODES_NUM):
        for j in range(NODES_NUM):
            index.append((i, j, i, j))
    # for e1 in edges1:
    #     for e2 in edges2:
    #         index.append((e1[0], e2[0], e1[1], e2[1]))
    #         index.append((e1[1], e2[1], e1[0], e2[0]))
    #         index.append((e1[0], e2[1], e1[1], e2[0]))
    #         index.append((e1[1], e2[0], e1[0], e2[1]))
    affinity = torch.sparse_coo_tensor(torch.tensor(index).t(), torch.ones(len(index)), (NODES_NUM, NODES_NUM, NODES_NUM, NODES_NUM))

    result = costomizedInterface(G, g, affinity, None)
    print(torch.equal(torch.tensor(result), M_gt))
    # print(G)
    # print(g)
    # print(affinity)
    # print(result)
    # print(M_gt)