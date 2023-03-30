#!/usr/bin/python

import pandas as pd 
import numpy as np
import re,os,sys
from GSVA import gsva, gmt_to_dataframe
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import roc_auc_score,roc_curve,auc,confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import cohen_kappa_score
import pickle
import argparse


def calTPM(data,length):
    tmp = pd.merge(data,length,left_index=True,right_index=True,how='inner')
    tmp = tmp.loc[:,tmp.columns[:-1]].apply(lambda x:x/tmp['length']).apply(lambda x:x* 1000000/ np.sum(x))
    return tmp

TrainAD =  pd.read_csv('LUAD.TCGA.coding.common.count',sep="\t",index_col=0)
TrainADlist = open('TCGA.LUAD.list').read().split('\n')
TrainADlist.pop()

TrainADcount = TrainAD[TrainADlist[:60]]
TrainLCLCcount = pd.read_csv('TestLCLC.coding.common.count',sep="\t",index_col=0)
TrainSCLCfpkm = pd.read_csv('TestSCLC.coding.common.FPKM',sep="\t",index_col=0)
TestADcount = pd.read_csv('AD.coding.common.count',sep="\t",index_col=0)
TestLCLCcount = pd.read_csv('TrainLCLC.coding.common.count',sep="\t",index_col=0)
TestSCLCfpkm = pd.read_csv('TestSCLC2.coding.common2.csv',sep="\t",index_col=0)

GL = pd.read_csv('GENELEN.txt',sep="\t",index_col=0)

TrainAD = calTPM(TrainADcount,GL)
TrainLCLC = calTPM(TrainLCLCcount,GL)
TrainSCLC = TrainSCLCfpkm.apply(lambda x: x * 1000000/ np.sum(x))
TestAD = calTPM(TestADcount,GL)
TestLCLC = calTPM(TestLCLCcount,GL)
TestSCLC = TestSCLCfpkm.apply(lambda x: x * 1000000/ np.sum(x))

Train = pd.merge(TrainLCLC,TrainSCLC,left_index=True,right_index=True,how='inner')
Train =  pd.merge(Train,TrainAD,left_index=True,right_index=True,how='inner')

Test = pd.merge(TestLCLC,TestSCLC,left_index=True,right_index=True,how='inner')
Test =  pd.merge(Test,TestAD,left_index=True,right_index=True,how='inner')

Train = Train.loc[Test.index]

### 重新标准化，因为丢失了一部分基因
Train = Train.apply(lambda x: x * 1000000/ np.sum(x)).apply(lambda x: np.log(x+1))
Test = Test.apply(lambda x: x * 1000000/ np.sum(x)).apply(lambda x: np.log(x+1))

genesets_df = gmt_to_dataframe('./c2.cp.kegg.v7.1.symbols.gmt')
Train_df = gsva(Train,genesets_df,method="gsva",tempdir="./")
Test_df = gsva(Test,genesets_df,method="gsva",tempdir="./")

Train_df = Train_df.T
Test_df = Test_df.T

Test_df['Label'] = 0
for i in Test_df.index:
    if 'LL' in i:
        Test_df.loc[i,'Label'] = 0
    elif  re.search(r'^S',i) and len(i) == 6:
        Test_df.loc[i,'Label'] = 1
    else:
        Test_df.loc[i,'Label'] = 2

Train_df['Label'] = 0
for i in Train_df.index:
    if 'LCNEC' in i:
        Train_df.loc[i,'Label'] = 0
    elif  re.match(r'^S',i):
        Train_df.loc[i,'Label'] = 1
    else:
        Train_df.loc[i,'Label'] = 2

Test_df['Label_LCLC'] = 1*(Test_df['Label'] == 0)
Test_df['Label_SCLC'] = 1*(Test_df['Label'] == 1)
Test_df['Label_LUAD'] = 1*(Test_df['Label']== 2)

Train_df['Label_LCLC'] = 1*(Train_df['Label'] == 0)
Train_df['Label_SCLC'] = 1*(Train_df['Label'] == 1)
Train_df['Label_LUAD'] = 1*(Train_df['Label']== 2)

model = OneVsRestClassifier(LogisticRegressionCV(Cs=np.logspace(-2, 2, 20), solver='liblinear',cv=10,penalty='l2',max_iter=5000,class_weight='balanced',random_state=123))
model.fit(Train_df.loc[:,Train_df.columns[:-4]],Train_df['Label'])
test_predict = model.predict_proba(Test_df.loc[:,Train_df.columns[:-4]])
test_pre = model.predict(Test_df.loc[:,Train_df.columns[:-4]])
test_auc = roc_auc_score(Test_df[['Label_LCLC','Label_SCLC','Label_LUAD']], test_predict,average='micro')
print (test_auc)
test_fpr, test_tpr, test_thresholds = roc_curve(np.array(Test_df[['Label_LCLC','Label_SCLC','Label_LUAD']]).ravel(), y_score=test_predict.ravel(), pos_label=1)
conf_mat = confusion_matrix(Test_df["Label"], test_pre)
conf_max_pro = pd.DataFrame(conf_mat).apply(lambda x:x/x.sum(),axis=1)
print (cohen_kappa_score(Test_df["Label"], test_pre))

Model = open('./LCNEC.pickle','wb')
pickle.dump(model,Model)
Model.close()
