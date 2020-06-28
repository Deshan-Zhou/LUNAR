from settings import *
import pandas as pd
from model import *
from optimizer import *
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import time
import os

parm = get_settings()

class DTIPredict:
    def __init__(self):
        #CUDA_VISIBLE_DEVICES="0" "0,1,2"  (nohup) python  your_file.py (>out.log 2>&1 &)
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        start = time.clock()

        self.load_data(parm.path)

        #init graph
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            #create a GCNAEModel
            self.model = GCNAEModel()

            #create optimizer
            self.op = Optimizer(
                reconstruct_loss=self.model.reconstruct_lost,
                reconstruct_result=self.model.reconstruct_result,
                all_atten_parm = self.model.all_atten_parm
            )

        #create data set and train
        self.create_set()

        end = time.clock()
        print('Running time: %s Hours' % ((end - start)/3600))

    def load_data(self, path):
        self.dg_dg = pd.read_csv(path+'drug_drug.csv',header=0,index_col=0)
        self.dg_pt = pd.read_csv(path+'drug_protein.csv',header=0,index_col=0)
        self.dg_ds = pd.read_csv(path+'drug_disease.csv',header=0,index_col=0)
        self.dg_se = pd.read_csv(path + 'drug_se.csv', header=0, index_col=0)
        self.dg_chem = pd.read_csv(path + 'drug_drug_similarity_in_chemical.csv', header=0, index_col=0)
        self.pt_dg = self.dg_pt.T
        self.pt_pt = pd.read_csv(path + 'protein_protein.csv', header=0, index_col=0)
        self.pt_ds = pd.read_csv(path + 'protein_disease.csv', header=0, index_col=0)
        self.pt_seq = pd.read_csv(path + 'protein_protein_similarity_in_sequence.csv', header=0, index_col=0)
        self.ds_dg = self.dg_ds.T
        self.ds_pt = self.pt_ds.T
        self.se_dg = self.dg_se.T

    def create_set(self):

        auc_set=[]
        aupr_set=[]

        for x in range(10):

            print("epoch:",str(x))
            whole_positive_index = []
            whole_negative_index = []
            for i in range(self.dg_pt.shape[0]):
                for j in range(self.dg_pt.shape[1]):
                    if int(self.dg_pt.iloc[i, j]) == 1:
                        whole_positive_index.append([i, j])
                    elif int(self.dg_pt.iloc[i, j]) == 0:
                        whole_negative_index.append([i, j])

            #NO. negative : NO. positive=10:1
            negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                         size=10 * len(whole_positive_index), replace=False)
            """
            #all data set
            negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                     size=len(whole_negative_index), replace=False)
            """

            data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)

            count = 0
            for i in whole_positive_index:
                data_set[count][0] = i[0]
                data_set[count][1] = i[1]
                data_set[count][2] = 1
                count += 1
            for i in negative_sample_index:
                data_set[count][0] = whole_negative_index[i][0]
                data_set[count][1] = whole_negative_index[i][1]
                data_set[count][2] = 0
                count += 1

            rs = np.random.randint(0, 1000, 1)[0]
            kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rs)
            for train_index, test_index in kf.split(data_set[:, 0:2], data_set[:, 2]):
                DTItrain, DTItest = data_set[train_index], data_set[test_index]
                DTItrain, DTIvalid = train_test_split(DTItrain, test_size=0.05, random_state=rs)
                v_auc, v_aupr, t_auc, t_aupr = self.train_model(train=DTItrain, valid=DTIvalid, test=DTItest)
                auc_set.append(t_auc)
                aupr_set.append(t_aupr)

        print('final auc:',np.mean(auc_set),'final aupr',np.mean(aupr_set))

        #save the repositioning network as results.csv
        final_results = pd.DataFrame(index=self.dg_pt._stat_axis.values,columns=self.dg_pt.columns.values,data=self.reconstruct_result)
        final_results.to_csv(parm.path + 'results.csv', encoding='utf-8')

        #save the auc and aupr
        data=[]
        data.append(np.mean(auc_set))
        data.append(np.mean(aupr_set))
        np.savetxt(parm.path+'results_auc_aupr.txt',data,encoding='utf-8-sig')
        #save the attention weight
        for i,atten_parms in self.all_atten_parms.items():
            if i == 0:
                for j in range(len(atten_parms)):
                    atten_result = pd.DataFrame(index=self.dg_pt._stat_axis.values, data=atten_parms[j])
                    if j == 0:
                        atten_result.to_csv(parm.path + 'dg_atten_dg.csv',encoding='utf-8')
                    if j == 1:
                        atten_result.to_csv(parm.path + 'dg_atten_pt.csv', encoding='utf-8')
                    if j == 2:
                        atten_result.to_csv(parm.path + 'dg_atten_ds.csv', encoding='utf-8')
                    if j == 3:
                        atten_result.to_csv(parm.path + 'dg_atten_se.csv', encoding='utf-8')
                    if j == 4:
                        atten_result.to_csv(parm.path + 'dg_atten_dgchem.csv', encoding='utf-8')
            if i == 1:
                for j in range(len(atten_parms)):
                    atten_result = pd.DataFrame(index=self.pt_dg._stat_axis.values, data=atten_parms[j])
                    if j == 0:
                        atten_result.to_csv(parm.path + 'pt_atten_dg.csv',encoding='utf-8')
                    if j == 1:
                        atten_result.to_csv(parm.path + 'pt_atten_pt.csv',encoding='utf-8')
                    if j == 2:
                        atten_result.to_csv(parm.path + 'pt_atten_ds.csv',encoding='utf-8')
                    if j == 3:
                        atten_result.to_csv(parm.path + 'pt_atten_ptseq.csv',encoding='utf-8')
            if i == 2:
                for j in range(len(atten_parms)):
                    atten_result = pd.DataFrame(index=self.ds_dg._stat_axis.values, data=atten_parms[j])
                    if j == 0:
                        atten_result.to_csv(parm.path + 'ds_atten_dg.csv',encoding='utf-8')
                    if j == 1:
                        atten_result.to_csv(parm.path + 'ds_atten_pt.csv',encoding='utf-8')
            if i == 3:
                for j in range(len(atten_parms)):
                    atten_result = pd.DataFrame(index=self.se_dg._stat_axis.values, data=atten_parms[j])
                    if j == 0:
                        atten_result.to_csv(parm.path + 'se_atten_dg.csv', encoding='utf-8')

    def train_model(self,train,valid,test):

        #Initialize the drug-protein network
        dg_pt = np.zeros((self.dg_pt.shape[0],self.dg_pt.shape[1]))
        mask = np.zeros((self.dg_pt.shape[0],self.dg_pt.shape[1]))
        #The positive and negative examples of the training set are added to dg_pt, 1 represents positive examples
        #The 1 in the mask indicates the positive and negative examples of the training set
        for ele in train:
            dg_pt[ele[0],ele[1] ] = ele[2]
            mask[ele[0],ele[1]] = 1
        pt_dg = dg_pt.T

        best_valid_aupr = 0
        best_valid_auc = 0
        test_aupr = 0
        test_auc = 0

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(parm.step):
                _,reconstruct_loss,self.reconstruct_result,self.all_atten_parms = sess.run([self.op.optimizer1,self.op.reconstruct_loss,self.op.reconstruct_result,self.op.all_atten_parm],\
                                                                            feed_dict={self.model.adj_mat[(0,0)]:self.dg_dg,\
                                                                                      self.model.adj_mat[(0,1)]:dg_pt,\
                                                                                      self.model.adj_mat[(0,2)]:self.dg_ds,\
                                                                                      self.model.adj_mat[(0,3)]:self.dg_se,\
                                                                                      self.model.adj_mat[(0,4)]:self.dg_chem,\
                                                                                      self.model.adj_mat[(1,0)]:pt_dg,\
                                                                                      self.model.adj_mat[(1,1)]:self.pt_pt,\
                                                                                      self.model.adj_mat[(1,2)]:self.pt_ds,\
                                                                                      self.model.adj_mat[(1,4)]:self.pt_seq,\
                                                                                      self.model.adj_mat[(2,0)]:self.ds_dg,\
                                                                                      self.model.adj_mat[(2,1)]:self.ds_pt,\
                                                                                      self.model.adj_mat[(3,0)]:self.se_dg,\
                                                                                      self.model.dg_pt_mask:mask})

                if i % 25 == 0:
                    print('step', i, 'loss',reconstruct_loss)

                    pred_list = []
                    ground_truth = []
                    for ele in valid:
                        pred_list.append(self.reconstruct_result[ele[0], ele[1]])
                        ground_truth.append(ele[2])
                    valid_auc = roc_auc_score(ground_truth, pred_list)
                    valid_aupr = average_precision_score(ground_truth, pred_list)

                    if valid_aupr >= best_valid_aupr:
                        best_valid_aupr = valid_aupr
                        best_valid_auc = valid_auc
                        pred_list = []
                        ground_truth = []
                        for ele in test:
                            pred_list.append(self.reconstruct_result[ele[0], ele[1]])
                            ground_truth.append(ele[2])
                        test_auc = roc_auc_score(ground_truth, pred_list)
                        test_aupr = average_precision_score(ground_truth, pred_list)
                    print('valid auc aupr,', valid_auc, valid_aupr, 'test auc aupr', test_auc, test_aupr)

        return best_valid_auc, best_valid_aupr, test_auc, test_aupr

dtipredict = DTIPredict()

