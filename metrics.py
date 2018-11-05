from utils import *
from utils import variable_summaries
import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    fin_loss=tf.reduce_mean(loss)
    variable_summaries(fin_loss)
    return fin_loss


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    accuracy=tf.reduce_mean(accuracy_all)
    variable_summaries(accuracy)
    return accuracy


def smoothness_reg(labels, graphs, loss, reg_scalar):
    for i in range(len(graphs)):
        # the first power of each graph in the list ...  maybe transform to Laplacian?
        pre_trace_tensor= reg_scalar*tf.matmul(tf.transpose(tf.sparse_tensor_dense_matmul( graphs[i][0],labels)), labels)
        pre_reg = tf.trace(pre_trace_tensor) / tf.cast(tf.shape(labels)[0] * tf.shape(labels)[1], 'float32')
        loss = loss+pre_reg

    return loss