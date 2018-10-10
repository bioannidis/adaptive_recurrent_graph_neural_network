from __future__ import division
from __future__ import print_function

import time
import os
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP, AGCN, AGRCN

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
def get_var_value(filename="varstore.dat"):
    with open(filename, "r+") as f:
        val = float(f.read() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(str(val))
        return val


def test_architecture(FLAGS,train_input,file):
    # Load data
    sys.stdout = file
    adj_list, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = train_input
    tf.reset_default_graph()

    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        supports = preprocess_adj_list(adj_list)
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        supports = chebyshev_polynomials(adj_list, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'agcn':
        supports = preprocess_adj_list(adj_list)
        num_supports = 1
        num_graphs = len(adj_list)
        model_func = AGCN
    elif FLAGS.model == 'agrcn':
        supports = preprocess_adj_list(adj_list)
        num_supports = 1
        num_graphs = len(adj_list)
        model_func = AGRCN
    elif FLAGS.model == 'agcn_cheby':
        supports = [chebyshev_polynomials(adj_list, FLAGS.max_degree)]
        num_supports = 1 + FLAGS.max_degree
        model_func = AGCN
    elif FLAGS.model == 'dense':
        supports = adj_list  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    with tf.name_scope('placeholders'):
        placeholders = {
            'supports': [[tf.sparse_placeholder(tf.float32, name='graph_' + str(i) + '_hop_' + str(k)) for k in
                          range(num_supports)] for i in range(num_graphs)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64),
                                              name='input_feautures'),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1]), name='labels'),
            'labels_mask': tf.placeholder(tf.int32, name='label_mask'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            'num_features_nonzero': tf.placeholder(tf.int32, name='num_features_nonzero')
        # helper variable for sparse dropout
        }

    # Create model

    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        # Define model evaluation function
        def evaluate(features, supports, labels, mask, placeholders):
            t_test = time.time()
            feed_dict_val = construct_feed_dict(features, supports, labels, mask, placeholders)
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            return outs_val[0], outs_val[1], (time.time() - t_test)

        # compare recursive with nonrecursive..... the graphs......

        # Init variables
        merged = tf.summary.merge_all()

        test_writer = tf.summary.FileWriter("/tmp/demo/2" + '/test')

        sess.run(tf.global_variables_initializer())

        cost_val = []
        writer = tf.summary.FileWriter("/tmp/demo/1")
        writer.add_graph(sess.graph)

        # Train model
        for epoch in range(FLAGS.epochs):
            train_writer = tf.summary.FileWriter("/tmp/demo/2" + '/train' + '/' + str(epoch),
                                                 sess.graph)
            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, supports, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            train_writer.add_summary(outs[0], epoch)
            # Validation
            cost, acc, duration = evaluate(features, supports, y_val, val_mask, placeholders)
            cost_val.append(cost)

            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[2]),
                  "train_acc=", "{:.5f}".format(outs[3]), "val_loss=", "{:.5f}".format(cost),
                  "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
                print("Early stopping...")
                break

        print("Optimization Finished!")

        # Testing
        test_cost, test_acc, test_duration = evaluate(features, supports, y_test, test_mask, placeholders)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
        file.close()
        sess.close()
        del_all_flags(FLAGS)
        return test_acc


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
learn_rates = [0.005]#np.linspace(0.01,0.08,4)
smooth_regs = np.logspace(-6,-4,3)
hidden_units = [64]#range(8,32,8)
dropout_rates = [0.2,0.6,0.85]#np.linspace(0.4,0.8,4)
sparse_regs = np.logspace(-7,-4,4)
weight_decays = np.logspace(-7,-4,4)
epochs=200
weight_decay=5e-4
model = 'agrcn'
neighbor_list=[2]
max_degree=3
sparse_reg=1e-4
early_stopping=50
dataset= 'cora'
your_counter = get_var_value()
folder_name= "results/tests"+str(your_counter)+"/"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
# Settings
test_results={}
with tf.device("/cpu:0"):
    train_input=load_data(dataset,neighbor_list)
with tf.device("/gpu:0"):
    for learn_rate in learn_rates:
        for smooth_reg in smooth_regs:
            for hidden_unit in hidden_units:
                for dropout_rate in dropout_rates:
                    for sparse_reg in sparse_regs:
                        for weight_decay in weight_decays:
                            flags = tf.app.flags
                            FLAGS = flags.FLAGS
                            flags.DEFINE_string('dataset', dataset, 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
                            flags.DEFINE_string('model', model, 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
                            flags.DEFINE_float('learning_rate', learn_rate, 'Initial learning rate.')
                            flags.DEFINE_integer('epochs', epochs, 'Number of epochs to train.')
                            flags.DEFINE_integer('hidden1', hidden_unit, 'Number of units in hidden layer 1.')
                            flags.DEFINE_float('dropout', dropout_rate, 'Dropout rate (1 - keep probability).')
                            flags.DEFINE_float('weight_decay', weight_decay, 'Weight for L2 loss on embedding matrix.')
                            flags.DEFINE_integer('early_stopping', early_stopping, 'Tolerance for early stopping (# of epochs).')
                            flags.DEFINE_list('neighbor_list',neighbor_list,'List of nearest neighbor graphs')
                            flags.DEFINE_integer('max_degree', max_degree, 'Maximum Chebyshev polynomial degree.')
                            flags.DEFINE_float('reg_scalar', smooth_reg, 'Initial learning rate.')
                            flags.DEFINE_float('sparse_reg', sparse_reg, 'Weight of sparsity regularizer.')
                            test_identifier="config:"+"learn_rate="+str(learn_rate)+",smooth_reg="+str(smooth_reg)+",hidden_units="\
                                            +str(hidden_unit)+"epochs="+str(epochs)+",dropout_rate="+str(dropout_rate)+\
                                     ",weight_decay="+str(weight_decay)+"early_stopping="+str(early_stopping)+",neighbor_list="+\
                                     str(neighbor_list)+",max_degree="+str(max_degree)+",sparse_reg="+str(sparse_reg)
                            f = open(folder_name+"config:"+test_identifier+".txt", "w+")
                            test_acc = test_architecture(FLAGS,train_input, f)
                            test_results[test_identifier]=test_acc
                            f_res=open(folder_name+"final_results_dataset="+dataset+"_neighbor_list="+str(neighbor_list)+".txt",'w')
                            f_res.write(str(test_results))

