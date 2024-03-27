import matlab.engine
import pickle
import torch
import time

ADDPATH_PATH = "./addPath.m"

# start matlab engine and add path
eng = matlab.engine.start_matlab("-logfile ./matlab.log")
eng.run(ADDPATH_PATH, nargout=0)

def getEdges(adjMat: torch.Tensor):
    upperTriangle = torch.triu(adjMat, diagonal=1).nonzero()
    sortedIndices = upperTriangle[upperTriangle[:, 1].argsort(stable=True)]
    reversedIndices = torch.flip(sortedIndices, [1])
    return torch.cat((sortedIndices, reversedIndices), dim = 0)

# Python interface for fgm
def customizedInterface(adjMat1: torch.Tensor, adjMat2: torch.Tensor, nodeAffinity: torch.Tensor, edgeAffinity: torch.Tensor, ct: list = None):

    # convert torch tensor to matlab double
    adjMat1, adjMat2 = matlab.double(adjMat1.tolist()), matlab.double(adjMat2.tolist())
    n1, n2 = adjMat1.size[0], adjMat2.size[0]

    # generate parameters
    par1 = eng.st("link", "cus", "val", adjMat1)
    par2 = eng.st("link", "cus", "val", adjMat2)

    # generate graph using fgm code
    pt1 = matlab.double([[0 for _ in range(n1)] for _ in range(2)])
    pt2 = matlab.double([[0 for _ in range(n2)] for _ in range(2)])
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

    asg = eng.fgmD(KP, KQ, ct, gphs, [], [])
    return(torch.tensor([list(r) for r in asg['X']]))

def loadData(dataFile):
    with open(dataFile, 'rb') as f:
        return pickle.load(f)
        
DIFFUTILS_DATA = "./data/diffutils.dat"
DIFFUTILS_MEDIUM_DATA = "./data/diffutils_medium.dat"

def calculate_recall_precision(standard, detection):
    standard_set = set([tuple(x.numpy()) for x in standard])
    detection_set = set([tuple(x.numpy()) for x in detection])

    tp = len(standard_set & detection_set)
    fp = len(detection_set - standard_set)
    fn = len(standard_set - detection_set)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    return recall, precision

if __name__ == "__main__":
    nodeAffinity, edgeAffinity, adjMat1, adjMat2, gt_r, gt_c = loadData(DIFFUTILS_MEDIUM_DATA)
    print(nodeAffinity.shape)
    print(edgeAffinity.shape)
    print(adjMat1.shape)
    print(adjMat2.shape)

    gt = torch.stack((torch.tensor(gt_r), gt_c), dim=0).t()

    result = customizedInterface(adjMat1, adjMat2, nodeAffinity, edgeAffinity).nonzero()

    print(result)
    print(calculate_recall_precision(gt, result))

    
    
