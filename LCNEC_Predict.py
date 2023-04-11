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


def Normalization_Expression(RNA_File,GL,RNA_list,Flag = False):
    Sample = pd.read_csv(RNA_File,sep="\t",index_col=0)
    if Flag:
        Sample = Sample.apply(lambda x: x * 1000000/ np.sum(x))
    else:
        Sample = calTPM(Sample,GL)
    
    Sample = Sample.loc[RNA_list].apply(lambda x: x * 1000000/ np.sum(x)).apply(lambda x: np.log(x+1))
    return Sample

def Predict_tumor(model,Sample,SoftwareDir,method="gsva"):
    genesets_df = gmt_to_dataframe(SoftwareDir+'/c2.cp.kegg.v7.1.symbols.gmt')
    Sample_gsva = gsva(Sample,genesets_df,method=method,tempdir= SoftwareDir+"/")
    Sample_gsva = Sample_gsva.T
    Sample_predict = model.predict_proba(Sample_gsva.loc[:,model.feature_names_in_])
    Output_Predict = pd.DataFrame(Sample_predict,columns=['Label_LCLC','Label_SCLC','Label_LUAD'],index=Sample_gsva.index)
    Sample_pre = model.predict(Sample_gsva.loc[:,model.feature_names_in_])
    Output_Pre = pd.DataFrame(Sample_pre,columns=['Label'],index=Sample_gsva.index)
    Output = pd.concat([Output_Predict,Output_Pre],axis=1)
    return Output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-R","--RNA",dest="rna",required=True, help="input RNA expression.GeneID is the index and separator is tabs")
    parser.add_argument("-F","--FPKM",dest="fpkm",default= False,required=False, action="store_true",help="RNA must be caculated by FPKM or Count; if RNA is FPKM, you can use this parameter")
    parser.add_argument("-M","--Method",dest="method",required=False, default= "gsva",help="the method of GSVA. If a single sample is tested, please use ssgsea. Detailed information can be read in the GSVA.R package")
    parser.add_argument("-D","--directory",dest="softdir",required=True, help="input the directory of software")
    parser.add_argument("-O","--outdir",dest="outdir",required=False,default="./", help="input the output directory")
    parser.add_argument("-G","--GSVA",dest="GSVA",required=False, action="store_true",help="use gsvascore to buile model")
    args = parser.parse_args()
    #print (args.fpkm)



    RNA_list = open(args.softdir+"/GENE.txt").read().split("\n")
    RNA_list.pop()
    GL = pd.read_csv(args.softdir+'/GENELEN.txt',sep="\t",index_col=0)
    
    Model = open(args.softdir+"/LCNEC.pickle","rb")
    model =  pickle.load(Model)
    Model.close
    
    Sample = Normalization_Expression(args.rna,GL,RNA_list,args.fpkm)
    
    if args.GSVA:
    
        Output = Predict_tumor(model,Sample,args.softdir,args.method)
        Output.to_csv(args.outdir+"/Tumor_Predict.txt",sep="\t",index=1)
    
    else:
    
        Sample_predict = model.predict_proba(Sample.T.loc[:,model.feature_names_in_])
        Output_Predict = pd.DataFrame(Sample_predict,columns=['Label_LCLC','Label_SCLC','Label_LUAD'],index=Sample.T.index)
        
        Sample_pre = model.predict(Sample.T.loc[:,model.feature_names_in_])
        Output_Pre = pd.DataFrame(Sample_pre,columns=['Label'],index=Sample.T.index)
        Output = pd.concat([Output_Predict,Output_Pre],axis=1)
        Output.to_csv(args.outdir+"/Tumor_Predict.txt",sep="\t",index=1)
        
if __name__ == "__main__":
    main()
