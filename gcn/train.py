from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP, AGCN, AGRCN

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'agrcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 12, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_list('neighbor_list',[5,10,20,40,60],'List of nearest neighbor graphs')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_float('reg_scalar', 1e-6, 'Initial learning rate.')
flags.DEFINE_float('sparse_reg', 1e-8, 'Weight of sparsity regularizer.')

# Load data
adj_list, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    supports= preprocess_adj_list(adj_list)
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    supports= chebyshev_polynomials(adj_list, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'agcn':
    supports= preprocess_adj_list(adj_list)
    num_supports = 1
    num_graphs = len(adj_list)
    model_func = AGCN
elif FLAGS.model == 'agrcn':
    supports = preprocess_adj_list(adj_list)
    num_supports = 1
    num_graphs = len(adj_list)
    model_func = AGRCN
elif FLAGS.model == 'agcn_cheby':
    supports= [chebyshev_polynomials(adj_list, FLAGS.max_degree)]
    num_supports = 1 + FLAGS.max_degree
    model_func = AGCN
elif FLAGS.model == 'dense':
    supports =adj_list # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
with tf.name_scope('placeholders'):
    placeholders = {
        'supports': [[tf.sparse_placeholder(tf.float32,name='graph_'+str(i)+'_hop_'+str(k)) for k in range(num_supports)] for i in range(num_graphs)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64), name='input_feautures'),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1]), name='labels'),
        'labels_mask': tf.placeholder(tf.int32,name= 'label_mask'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'num_features_nonzero': tf.placeholder(tf.int32,name='num_features_nonzero')  # helper variable for sparse dropout
    }

# Create model
#with tf.device("/gpu:0"):
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# Define model evaluation function
def evaluate(features, supports, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, supports, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

#compare recursive with nonrecursive..... the graphs......

# Init variables
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("/tmp/demo/2" + '/train',
                                      sess.graph)
test_writer = tf.summary.FileWriter("/tmp/demo/2" + '/test')

sess.run(tf.global_variables_initializer())

cost_val = []
writer = tf.summary.FileWriter("/tmp/demo/1")
writer.add_graph(sess.graph)

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, supports, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Training step
    outs = sess.run([merged,model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    train_writer.add_summary(outs[0],epoch)
    # Validation
    cost, acc, duration = evaluate(features, supports, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[2]),
          "train_acc=", "{:.5f}".format(outs[3]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, supports, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
