import tensorflow as tf
import numpy as np
import math
#global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """assigns unique layer IDs"""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def weight_variable_glorot(input_dim, output_dim, names=""):
    initial = tf.random.truncated_normal([int(input_dim),output_dim], stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial,name=names,dtype=tf.float32)


def row_pooling(adj_mat,flag):
    """Make adj_mat normalizing"""
    if flag == True:
        x = tf.diag_part(adj_mat)
        diag_matrix = tf.matrix_diag(x)
        adj_mat = adj_mat -diag_matrix

    adj_mat = tf.cast(adj_mat,dtype=tf.float32)
    small = tf.constant(value=1e-10,dtype=tf.float32,shape=adj_mat.get_shape().as_list())
    row_sums =tf.reduce_sum(adj_mat,axis=1)
    new_matrix = adj_mat / (row_sums[:, np.newaxis] + small)
    return new_matrix

class MultiLayer(object):
    """Defines base layer class for all layer objects"""
    def __init__(self,**kwargs):
        allowed_kwargs = {'name','logging'}
        for kwarg in kwargs.keys():
            #if kwargs belong to allowed_kwargs ,then show 'Invalid....',if not show exception
            assert kwarg in allowed_kwargs, 'Invalid keyword argument:' + kwarg
        name = kwargs.get('name')
        if not name:
            #return the name of the class who inherit the MultiLayer
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging',False)
        self.logging = logging

    def _call(self,inputs):
        return inputs

    #make a class instance like a fuc who can accept parm
    def __call__(self,inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

class GraphConvolutionMulti(MultiLayer):
    """Basic gragh convolution layer for undirected graph without edge labels"""
    def __init__(self, input_dim, output_dim, adj_mats,flag,dropout=0.,act=tf.nn.relu, **kwargs):
        super(GraphConvolutionMulti, self).__init__(**kwargs)
        self.adj_mats = adj_mats
        self.dropout = dropout
        self.act = act
        self.flag = flag
        #print('Weighted namespace：',self.name)
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['weight'] = weight_variable_glorot(
                input_dim, output_dim, names='weight')
            tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.vars['weight']))

    def _call(self, inputs):
        x = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.matmul(x, self.vars['weight'])
        x = tf.matmul(row_pooling(self.adj_mats,self.flag), x)
        outputs = self.act(x)
        return outputs

class ProjectReconstructionDecoder(MultiLayer):
    def __init__(self,emb1,emb2,input_dim,output_dim,mask = None,**kwargs):
        super(ProjectReconstructionDecoder,self).__init__(**kwargs)
        self.emb1 = emb1
        self.emb2 = emb2
        self.mask = mask
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['weight1'] = weight_variable_glorot(
                input_dim,output_dim,names='weight1'
            )
            tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.vars['weight1']))
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['weight2'] = weight_variable_glorot(
                input_dim,output_dim,names='weight2'
            )
            tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.vars['weight2']))

    def _call(self,adj_mats):
        x1 = tf.matmul(self.emb1,self.vars['weight1'])
        x2 = tf.matmul(self.emb2,self.vars['weight2'])
        x = tf.matmul(x1,x2,transpose_b=True)
        if self.mask is None:
            outputs = tf.reduce_sum(tf.multiply(adj_mats - x, adj_mats - x))
        else:
            temp = tf.multiply(self.mask,(adj_mats - x))
            outputs = tf.reduce_sum(tf.multiply(temp, temp))
        return x,outputs
