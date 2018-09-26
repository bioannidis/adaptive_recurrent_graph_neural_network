import sys
import networkx

def generate_data(nbr_nodes, nbr_of_features, graph_flags, graph_feat_flags, graph_label_flags):
    graph=generate_graph(nbr_nodes,graph_flags)
    features=generate_features(nbr_of_features, graph, graph_feat_flags)
    labels=generate_labels(graph,graph_label_flags)




def generate_graph(nbr_nodes,graph_flags):
    if graph_flags.erdos==1 :
        graph=networkx.fast_gnp_random_graph(n=nbr_nodes,p=graph_flags.edge_prop,directed=graph_flags.directed)

def generate_labels(graph, graph_label_flags):
    gr_adj = graph.adjacency()

def generate_features(nbr_of_features, graph, graph_feat_flags):
    gr_adj = graph.adjacency()

def sample_data(val_pct, test_pct):


def save_to_file(objects,dataset_str):
