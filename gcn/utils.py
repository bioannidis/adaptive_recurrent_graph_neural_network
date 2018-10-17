import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from sklearn.neighbors import kneighbors_graph
from sklearn import svm
import time

import tensorflow as tf

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
def noise_power_from_snrdb(snrdb):
    return 1/10.0 ** (snrdb/10.0)

def add_noise2feat(x,snrdb):
    noise_power=sp.linalg.norm(x)*noise_power_from_snrdb(snrdb)
    noise = noise_power* np.random.normal(0, 1, (np.shape(x)[1]))
    return x+noise

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str,neighbor_list):

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

    if (dataset_str=='test') | (dataset_str=='breast_cancer') | (dataset_str=='ionosphere') \
            | (dataset_str == 'synthetic'):
        with open("data/ind.{}.test.index".format(dataset_str), 'rb') as f:
            if sys.version_info > (3, 0):
                test_idx_reorder=(pkl.load(f, encoding='latin1'))
            else:
                test_idx_reorder =(pkl.load(f))
        if dataset_str=='test':
            adj = nx.adjacency_matrix(graph)
        else:
            adj =[]
        features = sp.vstack((allx)).tolil()
        labels = np.vstack((ally))
        test_idx_range = np.sort(test_idx_reorder)
    else:
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        labels = np.vstack((ally, ty))
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))




    features[test_idx_reorder, :] = features[test_idx_range, :]
    nbr_neighbors=neighbor_list
    adj_list=np.append([adj],create_network_nearest_neighbor(features,nbr_neighbors))

    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    val_size= 100
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+val_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return adj_list, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def create_network_nearest_neighbor(features,nbr_neighbors):
    adjacency_list=[]
    for nbr_neighbor in nbr_neighbors:
        adjacency=kneighbors_graph(features,n_neighbors=nbr_neighbor)
        adjacency_list=np.append(adjacency_list,[adjacency])
    return adjacency_list
def svm_class(x,y,tx,ty):
    clf = svm.SVC(gamma='scale')
    clf.fit(x,y)
    hatty=clf.predict(tx)
    return  hatty

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj_list(adj_list):
    adj_list_n=[]
    for adj in adj_list:
        adj_list_n.append([preprocess_adj(adj)])
    #adj_list_n=np.expand_dims(adj_list_n, axis=0) # for uniform inputs
    return adj_list_n

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, supports, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['supports'][i][j]: supports[i][j] for i in range(len(supports)) for j in range(len(supports[i]))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)



def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

#def test_architecture(FLAGS,train_input):
#    # Load data
#    #sys.stdout = file
#    adj_list, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = train_input
#    tf.reset_default_graph()
#
#    # Some preprocessing
#    features = preprocess_features(features)
#    if FLAGS.model == 'gcn':
#        supports = preprocess_adj_list(adj_list)
#        num_supports = 1
#        model_func = GCN
#    elif FLAGS.model == 'gcn_cheby':
#        supports = chebyshev_polynomials(adj_list, FLAGS.max_degree)
#        num_supports = 1 + FLAGS.max_degree
#        model_func = GCN
#    elif FLAGS.model == 'agcn':
#        supports = preprocess_adj_list(adj_list)
#        num_supports = 1
#        num_graphs = len(adj_list)
#        model_func = AGCN
#    elif FLAGS.model == 'agrcn':
#        supports = preprocess_adj_list(adj_list)
#        num_supports = 1
#        num_graphs = len(adj_list)
#        model_func = AGRCN
#    elif FLAGS.model == 'agcn_cheby':
#        supports = [chebyshev_polynomials(adj_list, FLAGS.max_degree)]
#        num_supports = 1 + FLAGS.max_degree
#        model_func = AGCN
#    elif FLAGS.model == 'dense':
#        supports = adj_list  # Not used
#        num_supports = 1
#        model_func = MLP
#    else:
#        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
#
#    # Define placeholders
#    with tf.name_scope('placeholders'):
#        placeholders = {
#            'supports': [[tf.sparse_placeholder(tf.float32, name='graph_' + str(i) + '_hop_' + str(k)) for k in
#                          range(num_supports)] for i in range(num_graphs)],
#            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64),
#                                              name='input_feautures'),
#            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1]), name='labels'),
#            'labels_mask': tf.placeholder(tf.int32, name='label_mask'),
#            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
#            'num_features_nonzero': tf.placeholder(tf.int32, name='num_features_nonzero')
#        # helper variable for sparse dropout
#        }
#
#    # Create model
#
#    model = model_func(placeholders, input_dim=features[2][1], logging=True)
#
#    # Initialize session
#    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#
#        # Define model evaluation function
#        def evaluate(features, supports, labels, mask, placeholders):
#            t_test = time.time()
#            feed_dict_val = construct_feed_dict(features, supports, labels, mask, placeholders)
#            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
#            return outs_val[0], outs_val[1], (time.time() - t_test)
#
#        # compare recursive with nonrecursive..... the graphs......
#
#        # Init variables
#        merged = tf.summary.merge_all()
#        sess.run(tf.global_variables_initializer())
#
#        cost_val = []
#
#        # Train model
#        for epoch in range(FLAGS.epochs):
#
#            # Construct feed dictionary
#            feed_dict = construct_feed_dict(features, supports, y_train, train_mask, placeholders)
#            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
#            # Training step
#            outs = sess.run([merged, model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
#            # Validation
#            cost, acc, duration = evaluate(features, supports, y_val, val_mask, placeholders)
#            cost_val.append(cost)
#
#            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
#                print("Early stopping...")
#                break
#
#        print("Optimization Finished!")
#
#        # Testing
#        test_cost, test_acc, test_duration = evaluate(features, supports, y_test, test_mask, placeholders)
#        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
#              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
#        sess.close()
#        return test_acc
#
#