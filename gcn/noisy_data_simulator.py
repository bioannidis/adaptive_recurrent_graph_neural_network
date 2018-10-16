from __future__ import division
from __future__ import print_function

import time
import os
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP, AGCN, AGRCN

def add_feat_noise(features,snr):
    for ind in range(np.shape(features)[0]):
        features[ind, :] = add_noise2feat(features[ind, :], snrdb=snr)
    return features

def get_var_value(filename="varstore1.dat"):
    with open(filename, "r+") as f:
        val = float(f.read() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(str(val))
        return val



# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
learn_rates = [0.005]#np.linspace(0.01,0.08,4)
smooth_regs = np.logspace(-6,-5,3)
hidden_units1 = [64]#range(8,32,8)
hidden_units2 = [0]#range(8,32,8)
dropout_rates = [0.9]#np.linspace(0.4,0.8,4)
sparse_regs = np.logspace(-5,-3,3)
weight_decays = np.logspace(-6,-4,4)
snrs= [1, 10]
epochs=200
weight_decay=5e-4
model = 'agrcn'
neighbor_list=[2]
max_degree=3
sparse_reg=1e-4
early_stopping=50
dataset= 'ionosphere'
your_counter = get_var_value()
folder_name= "results/tests_noise"+str(your_counter)+"/"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
# Settings
monte_carlo = 3
test_results={}
with tf.device("/cpu:0"):
    adj_list, features, y_train, y_val, y_test, train_mask, val_mask, test_mask =load_data(dataset,neighbor_list)
with tf.device("/gpu:0"):
    for snr in snrs:
        features=add_feat_noise(features,snr=snr)
        train_input=adj_list, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
        for learn_rate in learn_rates:
            for smooth_reg in smooth_regs:
                for hidden_unit1 in hidden_units1:
                    for hidden_unit2 in hidden_units2:
                        for dropout_rate in dropout_rates:
                            for sparse_reg in sparse_regs:
                                for weight_decay in weight_decays:
                                    test_acc=np.zeros(shape=(monte_carlo,1))
                                    flags = tf.app.flags
                                    FLAGS = flags.FLAGS
                                    flags.DEFINE_string('dataset', dataset, 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
                                    flags.DEFINE_string('model', model, 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
                                    flags.DEFINE_float('learning_rate', learn_rate, 'Initial learning rate.')
                                    flags.DEFINE_integer('epochs', epochs, 'Number of epochs to train.')
                                    flags.DEFINE_integer('hidden1', hidden_unit1, 'Number of units in hidden layer 1.')
                                    flags.DEFINE_integer('hidden2', hidden_unit2, 'Number of units in hidden layer 1.')
                                    flags.DEFINE_float('dropout', dropout_rate, 'Dropout rate (1 - keep probability).')
                                    flags.DEFINE_float('weight_decay', weight_decay, 'Weight for L2 loss on embedding matrix.')
                                    flags.DEFINE_integer('early_stopping', early_stopping, 'Tolerance for early stopping (# of epochs).')
                                    flags.DEFINE_list('neighbor_list',neighbor_list,'List of nearest neighbor graphs')
                                    flags.DEFINE_integer('max_degree', max_degree, 'Maximum Chebyshev polynomial degree.')
                                    flags.DEFINE_float('reg_scalar', smooth_reg, 'Initial learning rate.')
                                    flags.DEFINE_float('sparse_reg', sparse_reg, 'Weight of sparsity regularizer.')
                                    test_identifier="config:"+"learn_rate="+str(learn_rate)+",smooth_reg="+str(smooth_reg)+",hidden_units1="\
                                                    +str(hidden_unit1)+ "hidden_units2="\
                                                    +str(hidden_unit2)+"epochs="+str(epochs)+",dropout_rate="+str(dropout_rate)+\
                                             ",weight_decay="+str(weight_decay)+"early_stopping="+str(early_stopping)+",neighbor_list="+\
                                             str(neighbor_list)+",max_degree="+str(max_degree)+",sparse_reg="+str(sparse_reg)
                                    #f = open(folder_name+"config:"+test_identifier+".txt", "w+")
                                    for s_ind in range(monte_carlo):
                                        test_acc[s_ind] = test_architecture(FLAGS,train_input)
                                    del_all_flags(FLAGS)
                                    test_results[test_identifier]=np.mean(test_acc)
                                    f_res=open(folder_name+"final_results,snr="+snr+",dataset="+dataset+",neighbor_list="+str(neighbor_list)+".txt",'w')
                                    f_res.write(str(test_results))

