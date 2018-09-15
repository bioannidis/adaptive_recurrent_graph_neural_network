from gcn.inits import *
from gcn.utils import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                # glorot initializes the weights
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                # implements the input x matrix multiplication with the weights and has an option for sparse inputs
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            # implements multiplication with the graph i
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class AdaptiveGraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(AdaptiveGraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.graphs = placeholders['supports']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['graph_mixing_weight']=glorot([len(self.graphs), 1],
                                                        name='graph_mixing_weight')
            variable_summaries(self.vars['graph_mixing_weight'])
            for i in range(len(self.graphs)):
                # glorot initializes the weights
                for k in range(len(self.graphs[i])):
                    self.vars['weights_' + str(i)+'_'+str(k)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i)+'_'+str(k))
                    variable_summaries(self.vars['weights_' + str(i)+'_'+str(k)])
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
                variable_summaries(self.vars['bias'] )
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # input should include x and h maybe as a part of self?
        x = inputs
        with tf.name_scope("dropout"):
            # dropout
            if self.sparse_inputs:
                x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            else:
                x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        graph_outputs = list()
        # implement for loop for different graphs maybe supports variable

        for i in range(len(self.graphs)):
            with tf.name_scope("per_graph_proc"):
                graph=self.graphs[i]
                lin_combinations = list()
                # implement the k-hop neighborhood aggregation
                for k in range(len(graph)):
                    with tf.name_scope("feat_mix"):
                        if not self.featureless:
                            # implements the input x matrix multiplication with the weights and has an option for sparse inputs
                            pre_sup = dot(x, self.vars['weights_' + str(i)+'_' + str(k)],
                                          sparse=self.sparse_inputs)
                        else:
                            pre_sup = self.vars['weights_' + str(i)+'_'+str(k)]
                    # implements multiplication with the graph hop k
                    with tf.name_scope("diffuse_feat"):
                        graph_prod = dot(graph[k], pre_sup, sparse=True)
                        lin_combinations.append(graph_prod)
                # combines different hops
                with tf.name_scope("hops_mix"):
                    graph_output=tf.add_n(lin_combinations)
                    graph_outputs.append(graph_output)
        with tf.name_scope("combine_graph_outputs"):
            # implement mixing of the different graphs
            output =tf.squeeze(tf.tensordot(self.vars['graph_mixing_weight'],graph_outputs,[[0],[0]]))

        # bias
        if self.bias:
            output += self.vars['bias']

            # activation function on the output
        return self.act(output)


class AdaptiveGraphRecursiveConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, net_input, net_input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, first_layer=False, **kwargs):
        super(AdaptiveGraphRecursiveConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.graphs = placeholders['supports']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.first_layer = first_layer
        self.bias = bias
        self.net_input = net_input
        self.net_input_dim = net_input_dim

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        with tf.variable_scope(self.name + '_vars'):
            self.vars['graph_mixing_weight']=glorot([len(self.graphs), 1],
                                                        name='graph_mixing_weight')
            variable_summaries(self.vars['graph_mixing_weight'])
            for i in range(len(self.graphs)):
                # glorot initializes the weights
                for k in range(len(self.graphs[i])):
                    self.vars['weights_' + str(i)+'_'+str(k)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i)+'_'+str(k))
                    variable_summaries(self.vars['weights_' + str(i)+'_'+str(k)])
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
                variable_summaries(self.vars['bias'])
            if ~self.first_layer:
                self.vars['inp_graph_mixing_weight'] = glorot([len(self.graphs), 1],
                                                          name='inp_graph_mixing_weight')
                variable_summaries(self.vars['inp_graph_mixing_weight'])
                for i in range(len(self.graphs)):
                    # glorot initializes the weights
                    for k in range(len(self.graphs[i])):
                        self.vars['inp_weights_' + str(i) + '_' + str(k)] = glorot([net_input_dim, output_dim],
                                                                                   name='inp_weights_' + str(i) + '_' + str(k))
                        variable_summaries(self.vars['inp_weights_' + str(i) + '_' + str(k)])
                if self.bias:
                    self.vars['inp_bias'] = zeros([output_dim], name='inp_bias')
                    variable_summaries(self.vars['inp_bias'])

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # input should include h and h maybe as a part of self?
        # inputs should be the hidden inputs?
        x = self.net_input
        h = inputs

        with tf.name_scope("dropout"):
            # dropout
            if self.sparse_inputs:
                h = sparse_dropout(h, 1-self.dropout, self.num_features_nonzero)
                x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
            else:
                h = tf.nn.dropout(h, 1-self.dropout)
                x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)

        # convolve
        graph_outputs = list()
        inp_graph_outputs = list()
        # implement for loop for different graphs maybe supports variable

        for i in range(len(self.graphs)):
            with tf.name_scope("per_graph_proc"):
                graph=self.graphs[i]
                lin_combinations = list()
                inp_lin_combinations = list()

                # implement the k-hop neighborhood aggregation
                for k in range(len(graph)):
                    with tf.name_scope("feat_mix"):
                        if not self.featureless:
                            # implements the input h matrix multiplication with the weights and has an option for sparse inputs
                            pre_sup = dot(h, self.vars['weights_' + str(i)+'_' + str(k)],
                                          sparse=self.sparse_inputs)
                            inp_pre_sup = dot(x, self.vars['inp_weights_' + str(i) + '_' + str(k)],
                                          sparse=True)
                        else:
                            pre_sup = self.vars['weights_' + str(i)+'_'+str(k)]
                            inp_pre_sup = self.vars['inp_weights_' + str(i)+'_'+str(k)]
                    # implements multiplication with the graph hop k
                    with tf.name_scope("diffuse_feat"):
                        graph_prod = dot(graph[k], pre_sup, sparse=True)
                        lin_combinations.append(graph_prod)
                        inp_graph_prod = dot(graph[k], inp_pre_sup, sparse=True)
                        inp_lin_combinations.append(inp_graph_prod)
                # combines different hops
                with tf.name_scope("hops_mix"):
                    graph_output=tf.add_n(lin_combinations)
                    graph_outputs.append(graph_output)
                    inp_graph_output = tf.add_n(inp_lin_combinations)
                    inp_graph_outputs.append(inp_graph_output)
        with tf.name_scope("combine_graph_outputs"):
            # implement mixing of the different graphs
            hid_output = tf.squeeze(tf.tensordot(self.vars['graph_mixing_weight'],graph_outputs,[[0],[0]]))
            inp_output = tf.squeeze(tf.tensordot(self.vars['inp_graph_mixing_weight'], inp_graph_outputs, [[0], [0]]))
            output = tf.add(hid_output,inp_output,name='add_hidandfeats')

        # bias
        if self.bias:
            output += self.vars['bias']

            # activation function on the output
        return self.act(output)

