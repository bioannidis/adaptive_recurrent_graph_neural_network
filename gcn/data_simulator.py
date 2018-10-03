import sys
import networkx as nx
import random
import numpy as np
import  pickle as pkl
from scipy import  sparse


def lfr_reader_for_simple(folder_name):
    communities_full = np.genfromtxt(folder_name+'/community.dat', dtype=float)
    communities=communities_full[:,1]
    gr = nx.read_weighted_edgelist(folder_name+'/network.dat')
    return communities, gr
def one_hot_encoding(labels):
    nbr_labels=len(np.unique(labels))
    one_hot=np.zeros(shape=(len(labels),nbr_labels))
    for ind in range(len(labels)):
        one_hot[ind,int(labels[ind])-1]=1
    return one_hot

def generate_data(graph_flags, test_flags, graph_feat_flags, graph_label_flags,dataset_str):
    graph=generate_graph(graph_flags)
    features=generate_features(graph, graph_feat_flags, graph_flags)
    labels=generate_labels(graph,graph_label_flags,graph_flags)
    x, tx, y, ty, test_idx= sample_data(test_flags,labels,features)
    allx= sparse.csr_matrix(features)
    ally=labels
    save(x, y, tx, ty, allx, ally, graph, test_idx, dataset_str)


def generate_graph(graph_flags):
    if graph_flags['nograph']== 1:
        graph=[]
    elif graph_flags['erdos']['bool']==1 :
        graph=nx.fast_gnp_random_graph(n=graph_flags['nbr_nodes'],p=graph_flags.erdos.edge_prop,directed=graph_flags.directed)
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
    if graph_flags['breast_cancer']['bool'] == 1:
        features, labels = read_breast_cancer_data()
    elif graph_flags['adult']['bool'] == 1:
        features, labels = read_adult_data()
    elif graph_flags['erdos']['bool'] == 1:
        labels=None
    elif graph_flags['lfr']['bool'] == 1:
        labels, graph = lfr_reader_for_simple(graph_flags['lfr']['folder'])
    labels=one_hot_encoding(labels)
    return labels


def generate_features(graph, graph_feat_flags,graph_flags):
    if graph_feat_flags['featureless']== 1:
        features= np.identity(n=graph.number_of_nodes())
    elif graph_flags['breast_cancer']['bool'] == 1:
        features, labels = read_breast_cancer_data()
    elif graph_flags['adult']['bool'] == 1:
        features, labels = read_adult_data()
    elif graph_flags['erdos']['bool'] == 1:
        features = None
    elif graph_flags['lfr']['bool'] == 1:
        features, graph = lfr_reader_for_simple(graph_flags['lfr']['folder'])
    return features

def sample_data(test_flags,labels,features):
    bool_ind=np.ones((len(labels)),dtype=bool)
    start_test=(len(labels)-round(test_flags['test_pct']*len(labels)))
    end_train=round(test_flags['train_pct']*len(labels))
    bool_ind[start_test:-1] = False
    tx=(sparse.csr_matrix(features[~(bool_ind)]))
    ty=labels[(~bool_ind)]
    bool_ind = np.ones((len(labels)), dtype=bool)
    bool_ind[0:end_train] = False
    y = labels[~(bool_ind)]
    x = (sparse.csr_matrix(features[~(bool_ind)]))
    test_idx=range(start_test,len(labels))
    return x,tx,y,ty,test_idx

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


nbr_nodes=400
test_pct=0.2
train_pct=0.8
tau1=3
tau2=1.5
mu=0.1
average_degree=5
min_community=30
max_community=None
dataset_str='breast_cancer'
graph_flags={}
test_flags={}
test_flags['test_pct']=test_pct
test_flags['train_pct']=train_pct
graph_flags['nbr_nodes']=nbr_nodes
graph_flags['erdos']={}
graph_flags['nograph'] = 1

graph_flags['erdos']['bool']=0
graph_flags['lfr']={}
graph_flags['lfr']['bool']=0
graph_flags['lfr']['tau1']=tau1
graph_flags['lfr']['tau2']=tau2
graph_flags['breast_cancer']={}
graph_flags['breast_cancer']['bool']=0
graph_flags['adult']={}
graph_flags['adult']['bool']=1

graph_flags['lfr']['mu']=mu
graph_flags['lfr']['folder']='/home/umhadmin/research/weighted_networks/gen_network/network300'
graph_flags['lfr']['average_degree']=average_degree
graph_flags['lfr']['min_community']=min_community
graph_flags['lfr']['max_community']=max_community



graph_feat_flags={}
graph_label_flags={}

graph_feat_flags['featureless']=0

generate_data(graph_flags, test_flags, graph_feat_flags, graph_label_flags,dataset_str)