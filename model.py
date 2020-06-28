import tensorflow as tf
from settings import *
from layers import *

parm = get_settings()

class Model(object):
    def __init__(self,**kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invilid keyword argument:' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument:' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """Wrapper for _build()"""
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name:var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

class GCNAEModel(Model):
    def __init__(self,**kwargs):
        super(GCNAEModel,self).__init__(**kwargs)

        #process the adj_mat.(0,2) stand for different edge types
        self.adj_mat = {}
        self.adj_mat[(0,0)] = tf.placeholder(tf.float32,[parm.num_dg,parm.num_dg])
        self.adj_mat[(0,1)] = tf.placeholder(tf.float32,[parm.num_dg,parm.num_pt])
        self.adj_mat[(0,2)] = tf.placeholder(tf.float32,[parm.num_dg,parm.num_ds])
        self.adj_mat[(0,3)] = tf.placeholder(tf.float32,[parm.num_dg,parm.num_se])
        self.adj_mat[(0,4)] = tf.placeholder(tf.float32,[parm.num_dg,parm.num_dg])   #dg_chem
        self.adj_mat[(1,0)] = tf.placeholder(tf.float32,[parm.num_pt,parm.num_dg])
        self.adj_mat[(1,1)] = tf.placeholder(tf.float32,[parm.num_pt,parm.num_pt])
        self.adj_mat[(1,2)] = tf.placeholder(tf.float32,[parm.num_pt,parm.num_ds])
        self.adj_mat[(1,4)] = tf.placeholder(tf.float32,[parm.num_pt,parm.num_pt])   #pt_seq
        self.adj_mat[(2,0)] = tf.placeholder(tf.float32,[parm.num_ds,parm.num_dg])
        self.adj_mat[(2,1)] = tf.placeholder(tf.float32,[parm.num_ds,parm.num_pt])
        self.adj_mat[(3,0)] = tf.placeholder(tf.float32,[parm.num_se,parm.num_dg])

        #train set'location
        self.dg_pt_mask = tf.placeholder(tf.float32,[parm.num_dg,parm.num_pt])

        #initialize for embedding
        self.init_emb = {}
        init_dg_emb = tf.Variable(tf.truncated_normal(shape=[parm.num_dg, parm.dim_dg], stddev=0.1, dtype=tf.float32), dtype=tf.float32)
        init_pt_emb = tf.Variable(tf.truncated_normal(shape=[parm.num_pt, parm.dim_dg], stddev=0.1, dtype=tf.float32), dtype=tf.float32)
        init_ds_emb = tf.Variable(tf.truncated_normal(shape=[parm.num_ds, parm.dim_dg], stddev=0.1, dtype=tf.float32), dtype=tf.float32)
        init_se_emb = tf.Variable(tf.truncated_normal(shape=[parm.num_se, parm.dim_dg], stddev=0.1, dtype=tf.float32), dtype=tf.float32)
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(init_dg_emb))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(init_pt_emb))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(init_ds_emb))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(init_se_emb))
        self.init_emb[0] = init_dg_emb
        self.init_emb[1] = init_pt_emb
        self.init_emb[2] = init_ds_emb
        self.init_emb[3] = init_se_emb

        self.build()

    def _build(self):
        hidden1 = {}
        for i, j in self.adj_mat.keys():
            if i ==j:
                flag = True
            else:
                flag = False

            if i==0 and j==4:
                a = 0
            elif i==1 and j==4:
                a = 1
            else:
                a = j

            hidden1.setdefault(i,[]).append(
                GraphConvolutionMulti(
                input_dim=self.init_emb[a].shape[1], output_dim=parm.dim_hid1,
                adj_mats=self.adj_mat[(i,j)], act=lambda x: x,flag=flag,
                dropout=parm.dropout, logging=self.logging)(self.init_emb[a])
            )

        for i, hid1 in hidden1.items():
            W1 = weight_variable_glorot(input_dim=self.init_emb[i].shape[1], output_dim=parm.dim_hid1,
                                   names='hid1_weight' + str(i))
            tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W1))
            self_node = tf.matmul(self.init_emb[i],W1)
            hidden1[i] = tf.nn.relu(tf.add_n(hid1) + self_node)
            hidden1[i] = tf.nn.l2_normalize(hidden1[i], axis=1)


        hidden2 = {}
        for i, j in self.adj_mat.keys():
            if i ==j:
                flag = True
            else:
                flag = False

            if i==0 and j==4:
                a = 0
            elif i==1 and j==4:
                a = 1
            else:
                a = j

            hidden2.setdefault(i,[]).append(
                GraphConvolutionMulti(
                input_dim=parm.dim_hid1, output_dim=parm.dim_hid2,
                adj_mats=self.adj_mat[(i,j)], act=lambda x: x,flag=flag,
                dropout=parm.dropout, logging=self.logging)(hidden1[a])
            )

        for i, hid2 in hidden2.items():
            W2 = weight_variable_glorot(input_dim=parm.dim_hid1, output_dim=parm.dim_hid2, names='hid2_weight' + str(i))
            tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W2))
            self_node = tf.matmul(hidden1[i], W2)
            hidden2[i] = tf.nn.relu(tf.add_n(hid2) + self_node)
            hidden2[i] = tf.nn.l2_normalize(hidden2[i], axis=1)


        hidden3 = {}
        for i, j in self.adj_mat.keys():
            if i == j:
                flag = True
            else:
                flag = False

            if i == 0 and j == 4:
                a = 0
            elif i == 1 and j == 4:
                a = 1
            else:
                a = j

            hidden3.setdefault(i, []).append(
                GraphConvolutionMulti(
                    input_dim=parm.dim_hid2, output_dim=parm.dim_hid3,
                    adj_mats=self.adj_mat[(i, j)], act=lambda x: x, flag=flag,
                    dropout=parm.dropout, logging=self.logging)(hidden2[a])
            )

        """no attention mechanism
        self.embeddings = {}
        for i, embeds in self.hidden3.items():
            W = weight_variable_glorot(input_dim=parm.dim_hid2, output_dim=parm.dim_hid3, names='emb_weight' + str(i))
            tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W))
            self_node = tf.matmul(self.hidden2[i], W)
            self.embeddings[i] = tf.nn.relu(tf.add_n(embeds) + self_node)
            self.embeddings[i] = tf.nn.l2_normalize(self.embeddings[i], axis=1)
        """
        #attention mechanism
        self.embeddings = {}
        self.all_atten_parm = {}
        for i, embeds in hidden3.items():
            W = weight_variable_glorot(input_dim=parm.dim_hid2, output_dim=parm.dim_hid3, names='emb_weight' + str(i))
            tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W))
            self_node = tf.matmul(hidden2[i], W)

            atten_temp = []
            atten_final = []
            w = weight_variable_glorot(input_dim=parm.dim_hid3, output_dim=parm.dim_hid3, names='atten_weight' + str(i))
            b = tf.Variable(tf.constant(0.01, shape=[parm.dim_hid3], dtype=tf.float32))
            tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(w))
            tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(b))
            for embed in embeds:
                atten_temp.append(tf.exp(tf.nn.tanh(tf.matmul(embed,w) + b)))

            for j in range(len(embeds)):
                p = tf.divide(atten_temp[j], tf.add_n(atten_temp))
                self.all_atten_parm.setdefault(i,[]).append(p)
                atten_final.append(len(embeds)*tf.multiply(p,embeds[j]))

            self.embeddings[i] = tf.nn.relu(tf.add_n(atten_final) + self_node)
            self.embeddings[i] = tf.nn.l2_normalize(self.embeddings[i], axis=1)

        for i,j in self.adj_mat.keys():
            if (i == 1 and j == 0) or (i==2 and j==0) or (i==3 and j==0) or (i==2 and j==1):
                pass
            else:
                if i == 0 and j == 4:
                    a = 0
                elif i == 1 and j == 4:
                    a = 1
                else:
                    a = j
                if i==0 and j==0:
                    _,self.reconstruct_lost = ProjectReconstructionDecoder(
                        self.embeddings[i],self.embeddings[a],input_dim=parm.dim_hid3,output_dim=parm.dim_proj1,logging=self.logging
                    )(self.adj_mat[(i,j)])
                elif i==0 and j==1:
                    self.reconstruct_result,reconstruct_lost =ProjectReconstructionDecoder(
                        self.embeddings[i],self.embeddings[a],input_dim=parm.dim_hid3,output_dim=parm.dim_proj1,mask=self.dg_pt_mask,logging=self.logging
                    )(self.adj_mat[(i,j)])
                    self.reconstruct_lost += reconstruct_lost
                else:
                    _,reconstruct_lost = ProjectReconstructionDecoder(
                        self.embeddings[i],self.embeddings[a],input_dim=parm.dim_hid3,output_dim=parm.dim_proj1,logging=self.logging
                    )(self.adj_mat[(i,j)])
                    self.reconstruct_lost += reconstruct_lost

        self.l2_loss = tf.add_n(tf.get_collection("l2_reg"))
        self.reconstruct_lost += self.l2_loss


