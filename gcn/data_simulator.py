import sys
import networkx as nx
import random
import numpy as np
import  sklearn as sk
import  pickle as pkl
from scipy import  sparse
from sklearn import  datasets
import scipy.sparse as sp
from networkx.algorithms import community



def one_hot_encoding(labels):
    nbr_labels=len(np.unique(labels))
    one_hot=np.zeros(shape=(len(labels),nbr_labels))
    for ind in range(len(labels)):
        one_hot[ind,int(labels[ind])-1]=1
    return one_hot

def generate_data(graph_flags, test_flags, graph_feat_flags, graph_label_flags,dataset_str):
    graph=generate_graph(graph_flags)
    if graph_flags['random']['bool'] == 1:
        features,labels=generate_random_class_problem(graph_flags)
    else:
        features=generate_features(graph, graph_feat_flags, graph_flags)
        labels=generate_labels(graph,graph_label_flags,graph_flags)
    x, tx, y, ty, test_idx= sample_data(test_flags,labels,features)
    allx= sp.vstack((x, tx))
    ally = np.vstack((y, ty))
    save(x, y, tx, ty, allx, ally, graph, test_idx, dataset_str)


def generate_graph(graph_flags):
    if graph_flags['ionosphere']['bool']==1 | graph_flags['random']['bool'] == 1:
        graph=[]
    elif graph_flags['erdos']['bool']==1 :
        graph=nx.fast_gnp_random_graph(n=graph_flags['nbr_nodes'],p=graph_flags['erdos']['edge_prob'],
                                       directed=graph_flags['directed'])
    elif graph_flags['lfr']['bool']==1 :
        communities, graph= lfr_reader_for_simple(graph_flags['lfr']['folder'])
        # graph=nx.algorithms.community.community_generators\
        #     .LFR_benchmark_graph(graph_flags['nbr_nodes'], graph_flags['lfr']['tau1'],
        #                          graph_flags['lfr']['tau2'], graph_flags['lfr']['mu'],
        #                          average_degree=graph_flags['lfr']['average_degree'],
        #                          min_degree=None, max_degree=None,
        #                          max_community=graph_flags['lfr']['max_community'],
        #                          min_community=graph_flags['lfr']['min_community'], seed=None)
    return graph

def generate_labels(graph, graph_label_flags,graph_flags):
    if graph_flags['random']['bool'] == 1:
        features, labels = generate_random_class_problem(graph_flags)
    elif graph_flags['breast_cancer']['bool'] == 1:
        features, labels = read_breast_cancer_data()
    elif graph_flags['ionosphere']['bool'] == 1:
        features, labels = read_ionosphere_data()
    elif graph_flags['erdos']['bool'] == 1:
        communities_generator=community.girvan_newman(graph)
        top_level_communities=next(communities_generator)
        labels=None
    elif graph_flags['lfr']['bool'] == 1:
        com_labels, graph = lfr_reader_for_simple(graph_flags['lfr']['folder'])
        ban_labels=gen_ban_labels(graph)
        labels = np.expand_dims(ban_labels, axis=1) - 1
    return labels



def generate_features(graph, graph_feat_flags,graph_flags):
    if graph_flags['random']['bool'] == 1:
        features, labels = generate_random_class_problem(graph_flags)
    elif graph_flags['breast_cancer']['bool'] == 1:
        features, labels = read_breast_cancer_data()
    elif graph_flags['ionosphere']['bool'] == 1:
        features, labels = read_ionosphere_data()
    elif graph_flags['erdos']['bool'] == 1:
        features = np.identity(n=graph.number_of_nodes())
    elif graph_flags['lfr']['bool'] == 1:
        features = np.identity(n=graph.number_of_nodes())
    return features

def sample_data(test_flags,labels,features):
    # sample equal number exampl per class
    nbr_labels=len(np.unique(labels))
    train_ind=[]
    for ind in range(nbr_labels):
        itemindex = np.where(labels == ind)[0]
        train_ind =np.append(train_ind,itemindex[0:test_flags['nbr_exampl_per_class']])
        train_ind=train_ind.astype(int)
    labels=one_hot_encoding(labels)
    x=(sparse.csr_matrix(features[train_ind]))
    y=labels[(train_ind)]
    permute_ind = list(range(len(y)))
    np.random.shuffle(permute_ind)
    x = x[permute_ind]
    y = y[permute_ind]
    ty = labels[~np.in1d(range(len(labels)),train_ind)]
    tx = (sparse.csr_matrix(features[~np.in1d(range(len(labels)),train_ind)]))
    test_idx=range(len(train_ind),len(labels))
    return x,tx,y,ty,test_idx
def generate_random_class_problem(graph_flags):
    # gaussian with different variance and each gaussian corresponds to a label
    #X, y=datasets.make_classification(n_samples=graph_flags['nbr_nodes'],
    #                                n_classes=graph_flags['random']['nbr_classes'])
    iter=0
    for mean, var in zip( graph_flags['random']['means'], graph_flags['random']['vars']):
        v_mean=mean*np.ones(shape=(graph_flags['random']['dim'],1))
        v_var=var*np.ones(shape=(graph_flags['random']['dim'],1))
        if iter ==0:
            X= np.random.normal(v_mean, v_var, size=(graph_flags['random']['dim'],
                                                     graph_flags['random']['per_class_ex']))
            y = iter * np.ones( shape=(graph_flags['random']['per_class_ex'], 1))
        else:
            X =np.append(X,np.random.normal(v_mean,v_var, size=(graph_flags['random']['dim'],
                                             graph_flags['random']['per_class_ex'])), axis=1)
            y= np.append(y,iter*np.ones( shape=(graph_flags['random']['per_class_ex'],1)))
        iter+=1
    X=np.transpose(X)
    permute_ind = list(range(len(y)))
    np.random.shuffle(permute_ind)
    X = X[permute_ind]
    y = y[permute_ind]


    return X, y


def gen_ban_labels(graph):
    laplacian=nx.laplacian_matrix(graph)
    vals, vecs=sp.linalg.eigs(laplacian, k=2)
    vecs=np.real(vecs)
    ban_labels= ((vecs[:,1]>0)).astype(int)
    return ban_labels



def lfr_reader_for_simple(folder_name):
    # sys.path.append('/home/umhadmin/research/weighted_networks/')
    # call(["./benchmark",     "-f flags.dat"])
    communities_full = np.genfromtxt(folder_name+'/community.dat', dtype=float)
    communities=communities_full[:,1]
    gr = nx.read_weighted_edgelist(folder_name+'/network.dat')
    return communities, gr

def read_adult_data():
    data = []
    with open('/home/umhadmin/agrcn/gcn/raw_data/adult.data.txt', 'r') as inf:
        for line in inf:
            data.append((line))
    test1 = np.loadtxt(fname='/home/umhadmin/agrcn/gcn/raw_data/adult.test.txt',
                      delimiter=',')
    features=data[:,0:10]
    labels=data[:,-1]
    labels=labels/2-1
    return features,labels

def read_ionosphere_data():

    data = np.loadtxt(fname='/home/umhadmin/agrcn/gcn/raw_data/clasionosphere.data.txt',
                      delimiter=',')
    features = data[:, 0:33]
    labels = data[:, -1]
    return features,labels

def read_breast_cancer_data():
    data=np.loadtxt(fname='/home/umhadmin/agrcn/gcn/raw_data/breast-cancer-wisconsin.reduceddata.txt',
                    delimiter=',')
    features=data[:,0:10]
    labels=data[:,-1]
    labels=labels/2-1
    return features,labels
def save(x, y, tx, ty, allx, ally, graph,test_idx,dataset_str):
     with open("data/ind."+dataset_str+".x", 'wb+') as f:
         save_to_file(f,x)
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
     with open("data/ind." + dataset_str + ".graph", 'wb+') as f:
         save_to_file(f, graph)
     with open("data/ind." + dataset_str + ".test.index", 'wb+') as f:
         save_to_file(f, test_idx)




def save_to_file (file, obj):
    pkl.dump(obj,file=file)


nbr_nodes=1000
nbr_classes=2
tau1=3
tau2=1.5
variance=0.4
mu=0.1
average_degree=5
min_community=30
max_community=None
directed=False
nbr_exampl_per_class=50
edge_prob=0.05
dataset_str='ionosphere'
graph_flags={}
test_flags={}
graph_flags['nbr_nodes']=nbr_nodes
graph_flags['erdos']={}

graph_flags['erdos']['bool']=0
graph_flags['lfr']={}
graph_flags['random']={}
graph_flags['random']['bool']=0
graph_flags['random']['nbr_classes']=nbr_classes
graph_flags['random']['means']=np.linspace(0,1,num=nbr_classes)
graph_flags['random']['vars']=variance*np.ones(shape=(nbr_classes,1))
graph_flags['random']['dim']=10
graph_flags['random']['per_class_ex']=round(nbr_nodes/nbr_classes)

graph_flags['lfr']['bool']=0
graph_flags['lfr']['tau1']=tau1
graph_flags['lfr']['tau2']=tau2
graph_flags['breast_cancer']={}
graph_flags['breast_cancer']['bool']=0
graph_flags['adult']={}
graph_flags['adult']['bool']=0
graph_flags['erdos']['edge_prob']= edge_prob
graph_flags['ionosphere']= {}
graph_flags['ionosphere']['bool']=1
test_flags['nbr_exampl_per_class']=nbr_exampl_per_class
graph_flags['lfr']['mu']=mu
graph_flags['lfr']['folder']='/home/umhadmin/research/weighted_networks/gen_network/1network'+str(nbr_nodes)
graph_flags['lfr']['average_degree']=average_degree
graph_flags['lfr']['min_community']=min_community
graph_flags['lfr']['max_community']=max_community
graph_flags['directed']=directed


graph_feat_flags={}
graph_label_flags={}

graph_feat_flags['featureless']=0

generate_data(graph_flags, test_flags, graph_feat_flags, graph_label_flags,dataset_str)