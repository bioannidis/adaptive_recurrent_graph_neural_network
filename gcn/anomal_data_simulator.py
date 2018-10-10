import numpy as np
import pickle as pkl
import sys
from shutil import copyfile
def noise_power_from_snrdb(snrdb):
    return 1/10.0 ** (snrdb/10.0)

def add_noise2feat(x,snrdb):
    noise_power=noise_power_from_snrdb(snrdb)
    noise = noise_power* np.random.normal(0, 1, (np.shape(x)[1]))
    return x+noise

def create_corrupted_instances(pct,x,y,snr):
    y0=np.hstack((y,np.zeros((np.shape(y)[0],1))))
    nbr_cor_elem=round(pct*(np.shape(y)[0]))
    anomal_class=np.zeros((1,np.shape(y0)[1]))
    anomal_class[:,-1]=1
    for ind in range(nbr_cor_elem):
        y0[ind,:]=anomal_class
        x[ind,:]= add_noise2feat(x[ind,:],snrdb=snr)
    return  x,y0

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset_str):

    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    return  x, y, tx, ty, allx, ally, graph, test_idx_range
def store_data(init_dataset,dataset_str,x,y,tx,ty,allx,ally):
    with open("data/ind." + dataset_str + ".x", 'wb+') as f:
        save_to_file(f, x)
    with open("data/ind." + dataset_str + ".y", 'wb+') as f:
        save_to_file(f, y)
    with open("data/ind." + dataset_str + ".tx", 'wb+') as f:
        save_to_file(f, tx)
    with open("data/ind." + dataset_str + ".ty", 'wb+') as f:
        save_to_file(f, ty)
    with open("data/ind." + dataset_str + ".allx", 'wb+') as f:
        save_to_file(f, allx)
    with open("data/ind." + dataset_str + ".ally", 'wb+') as f:
        save_to_file(f, ally)
    copyfile("data/ind.{}.test.index".format(init_dataset), "data/ind.{}.test.index".format(dataset_str))
    copyfile("data/ind.{}.graph".format(init_dataset), "data/ind.{}.graph".format(dataset_str))

def save_to_file (file, obj):
    pkl.dump(obj,file=file)

dataset='cora'
new_dataset='cora+anomalies'
pct_train =0.1
pct_test=0.1
snr=0.1
x, y, tx, ty, allx, ally, graph, test_idx_range=load_data(dataset)
x,y=create_corrupted_instances(pct_train,x,y,snr=snr)
tx,ty=create_corrupted_instances(pct_train,tx,ty,snr=snr)
ally = np.hstack((ally, np.zeros((np.shape(ally)[0], 1))))
allx[0:np.shape(x)[0],:]=x
ally[0:np.shape(x)[0],:]=y

#allx[test_idx_range, :]=tx
#ally[test_idx_range, :]=ty
store_data(dataset,new_dataset, x, y, tx, ty, allx, ally)
