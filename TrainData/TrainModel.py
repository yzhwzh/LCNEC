import pandas as pd 
import numpy as np
import re,os,sys
from GSVA import gsva, gmt_to_dataframe
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import roc_auc_score,roc_curve,auc,confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA,NMF
import matplotlib.pyplot as plt
from matplotlib.ticker import  MultipleLocator
import matplotlib.colors as col
import matplotlib.cm as cm
import seaborn as sns
import pickle
import argparse

def calTPM(data,length):
    tmp = pd.merge(data,length,left_index=True,right_index=True,how='inner')
    tmp = tmp.loc[:,tmp.columns[:-1]].apply(lambda x:x/tmp['length']).apply(lambda x:x* 1000000/ np.sum(x))
    return tmp

def PCA_Analysis(Features,Frame,Label,outdir,name):
    
    Feat = list(set(Features).intersection(Frame.columns))
    
    Allpca = PCA(n_components=3)
    Allpca = Allpca.fit_transform(Frame[Feat])
    Allpca = pd.DataFrame(Allpca)
    Allpca.index = Frame.index
    Allpca['Label'] = Frame[Label]

    fig = plt.figure(figsize=(5,4),dpi=100)
    ax = fig.gca()
    font1 = {'size':7,'weight':'semibold',}
    ax = sns.scatterplot(x=0,y=1,data=Allpca,hue="Label",style="Label")
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(bbox_to_anchor=(1.01,0),loc="lower left",frameon=False,ncol=1,prop=font1,markerscale=1.3)
    ax.set_xlabel('The 1st Principal Component',fontsize=10)
    ax.set_ylabel('The 2nd Principal Component',fontsize=10)
    ax.tick_params(axis="both", which="both", labelsize=8, length=4, width=2)   
    ax.tick_params(axis="both", which="minor", labelsize=8, length=2, width=2) 
    #ax.yaxis.set_major_locator(MultipleLocator(5))
    #ax.yaxis.set_minor_locator(MultipleLocator(2.5))
    #ax.xaxis.set_major_locator(MultipleLocator(5))
    #ax.xaxis.set_minor_locator(MultipleLocator(2.5))  
    plt.savefig(outdir+'/'+name+'_PCA.svg')

def AUC_Analysis(Regular,Train,Test,Features,outdir,name):
    
    Feat = list(set(Features).intersection(Train.columns))
    
    model = OneVsRestClassifier(LogisticRegressionCV(Cs=np.logspace(-2, 2, 20), solver='liblinear',cv=10,penalty=Regular,max_iter=5000,class_weight='balanced',random_state=123))
    model.fit(Train.loc[:,Feat],Train['Label'])
    
    
    test_predict = model.predict_proba(Test.loc[:,Feat])
    test_pre = model.predict(Test.loc[:,Feat])
    test_auc = roc_auc_score(Test[['Label_LCLC','Label_SCLC','Label_LUAD']], test_predict,average='micro')
    test_fpr, test_tpr, test_thresholds = roc_curve(np.array(Test[['Label_LCLC','Label_SCLC','Label_LUAD']]).ravel(), y_score=test_predict.ravel(), pos_label=1)
    
    train_predict = model.predict_proba(Train.loc[:,Feat])
    train_auc = roc_auc_score(Train[['Label_LCLC','Label_SCLC','Label_LUAD']], train_predict,average='micro')
    train_fpr, train_tpr, train_thresholds = roc_curve(np.array(Train[['Label_LCLC','Label_SCLC','Label_LUAD']]).ravel(), y_score=train_predict.ravel(), pos_label=1)
    

    fig = plt.figure(figsize=(10,4),dpi=100)

    ax = plt.subplot(1,2,1)
    ax.plot(train_fpr,train_tpr,c='#3C5488FF',label='%s (AUC = %0.3f) '%("Train", train_auc),linewidth=1.5)
    ax.plot(test_fpr,test_tpr,c='#DC0000FF',label='%s (AUC = %0.3f) '%("Test", test_auc),linewidth=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.tick_params(axis="both", which="both", labelsize=10,length=4, width=2)
    ax.tick_params(axis="both", which="minor", length=2, width=2)
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.legend(loc='lower right',fontsize=10,frameon=False)   
    ax.plot([0,1],[0,1],linestyle='--',color='gray',linewidth=0.5)
    ax.set_xlim([-0.01,1.01])
    ax.set_ylim([-0.01,1.01])
    ax.set_ylabel("True Positive Rate",fontsize=15) 
    ax.set_xlabel('False Positive Rate',fontsize=15)
    ax.set_title('ROC Curve',fontsize=15)
    
    
    ax = plt.subplot(1,2,2)
    conf_mat = confusion_matrix(Test["Label"], test_pre)
    conf_max_pro = pd.DataFrame(conf_mat).apply(lambda x:x/x.sum(),axis=1)
    print ("kappa coefficient "+str(cohen_kappa_score(Test["Label"], test_pre)))
    #cmap = sns.diverging_palette(220, 10, as_cmap=True)
    ax = sns.heatmap(conf_max_pro, annot=conf_mat, fmt='d',xticklabels=['Label_LCLC','Label_SCLC','Label_LUAD'], yticklabels=['Label_LCLC','Label_SCLC','Label_LUAD'],square=True, linewidths=2,annot_kws={'fontsize':12,'fontweight':"semibold"})
    #ax = sns.heatmap(conf_max_pro, annot=True,xticklabels=['Label_LCLC','Label_SCLC','Label_LUAD'], yticklabels=['Label_LCLC','Label_SCLC','Label_LUAD'], cmap="own2",square=True, linewidths=2,annot_kws={'fontsize':12,'fontfamily':"serif",'fontweight':"semibold"})
    ax.set_ylabel('True Label',fontsize=12,fontfamily="serif",fontweight="semibold")
    ax.set_xlabel('Model Prediction Label',fontsize=12,fontfamily="serif",fontweight="semibold")
    ax.set_yticklabels(['Label_LCLC','Label_SCLC','Label_LUAD'],fontsize=8,fontfamily="serif",fontweight="semibold")
    ax.set_xticklabels(['Label_LCLC','Label_SCLC','Label_LUAD'],fontsize=8,fontfamily="serif",fontweight="semibold")
    ax.tick_params(axis="both", which="both",bottom=True,top=False,labelbottom=True, left=True, right=False, labelleft=True,labelsize=10)  
    ax.set_yticklabels(['Label_LCLC','Label_SCLC','Label_LUAD'],fontdict={'verticalalignment': 'center'})
      
    plt.savefig(outdir+'/'+name+'_ROC.svg')
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-I","--inputdir",dest="inputdir",required=True, default= ".",help="input the data directory")
    parser.add_argument("-O","--outdir",dest="outdir",required=False,default=".", help="input the output directory")
    parser.add_argument("-G","--GSVA",dest="gsva",required=False, action="store_true",help="use gsvascore to buile model")
    parser.add_argument("-P","--Prior",dest="prior",required=False,default="trans,cnv,mut", help="when gsva is setted False,you should choose a gene set as the prior gene set from cnv,trans,mut gene set,or you can use them together and join them by comma")
    parser.add_argument("-R","--Regular",dest="regular",required=True,default="l2", help="set the regular for model")
    args = parser.parse_args()
    
    
    TrainAD =  pd.read_csv(args.inputdir+'/LUAD.TCGA.coding.common.count',sep="\t",index_col=0)
    TrainADlist = open(args.inputdir+'/TCGA.LUAD.list').read().split('\n')
    TrainADlist.pop()

    TrainADcount = TrainAD[TrainADlist[:60]]
    TrainLCLCcount = pd.read_csv(args.inputdir+'/TestLCLC.coding.common.count',sep="\t",index_col=0)
    TrainSCLCfpkm = pd.read_csv(args.inputdir+'/TestSCLC.coding.common.FPKM',sep="\t",index_col=0)
    TestADcount = pd.read_csv(args.inputdir+'/AD.coding.common.count',sep="\t",index_col=0)
    TestLCLCcount = pd.read_csv(args.inputdir+'/TrainLCLC.coding.common.count',sep="\t",index_col=0)
    TestSCLCfpkm = pd.read_csv(args.inputdir+'/TestSCLC2.coding.common2.csv',sep="\t",index_col=0)

    GL = pd.read_csv(args.inputdir+'/GENELEN.txt',sep="\t",index_col=0)

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
    Train = Train.apply(lambda x: x * 1000000/ np.sum(x)).apply(lambda x: np.log(x+1)).T
    Test = Test.apply(lambda x: x * 1000000/ np.sum(x)).apply(lambda x: np.log(x+1)).T

    Test['Label'] = 0
    for i in Test.index:
        if 'LL' in i:
            Test.loc[i,'Label'] = 0
        elif  re.search(r'^S',i) and len(i) == 6:
            Test.loc[i,'Label'] = 1
        else:
            Test.loc[i,'Label'] = 2
    Test['Label2'] = Test['Label'].replace([0,1,2],["TestLCNEC","TestSCLC","TestLUAD"])                      
        
    Train['Label'] = 0
    for i in Train.index:
        if 'LCNEC' in i:
            Train.loc[i,'Label'] = 0
        elif  re.match(r'^S',i):
            Train.loc[i,'Label'] = 1
        else:
            Train.loc[i,'Label'] = 2
    Train['Label2'] = Train['Label'].replace([0,1,2],["TrainLCNEC","TrainSCLC","TrainLUAD"])

    Test['Label_LCLC'] = 1*(Test['Label'] == 0)
    Test['Label_SCLC'] = 1*(Test['Label'] == 1)
    Test['Label_LUAD'] = 1*(Test['Label']== 2)

    Train['Label_LCLC'] = 1*(Train['Label'] == 0)
    Train['Label_SCLC'] = 1*(Train['Label'] == 1)
    Train['Label_LUAD'] = 1*(Train['Label']== 2)
    
    trans =["AKR7A3","AMBP","ANXA13","ASCL1","CALCA","CCNE1","CDK19","CDK5R1","CDKN1B","CDKN2C","CDKN2D","CHGA","CREBBP","DLK1","DLL3","EFN4","ELAVL3","ELAVL4","EP300","EPHA2","GC","GFI1B","GRP","GUCY2C","HES6","HGD","HNF1A","HNF4A","IGFBP1","IHH","IL1R2","IL1RL2","ITGB4","JUB","MSH6","MYC","NCAM1","NEFM","NGB","NOTCH1","NOTCH2","NOTCH2NL","NOTCH3","NOTCH4","NPAS4","NR0B2","POLA1","PSIP1","PTEN","PVT1","RB1","RFX6","RUNX1","RUNX2","SCG3","SCGN","SOX1","SOX2","SYN","SYP","TGIF2","TLR2","TLR5","TNFRSF10B","TNFRSF1A","TP53","TP73","TRADD","YAP1"]
    mut = ["DAMTS12","ADAMTS2","ADCY1","ALK","ALMS1","ARID1A","ASPM","BCLAF1","BRAF","C17orf108","CDKN2A","CDYL","CNTNAP2","COBL","COL22A1","COL4A2","CREBBP","DDR2","DIP2C","EGFR","ELAVL2","EP300","EPHA7","ERBB2","FGFR1","FGFR3","FMN2","FPR1","GAS7","GRIK3","GRM8","IRS2","KEAP1","KHSRP","KIAA1211","KIF21A","KIT","KRAS","MET","MGA","MLL","MYC","MYCL1","NF1","NFE2L2","NKX2","NOTCH1","NOTCH2","NOTCH3","NOTCH4","NTM","PDE4DIP","PIK3CA","PLSCR4","PTEN","PTGFRN","PTPRD","RASSF8","RB1","RBL1","RBL2","RBM10","RGS7","RIMS2","RIT1","RUNX1T1","SATB2","SETD2","SLIT2","SMARCA4","STK11","TMEM132D","TP53","TP73","U2AF1","XRN1","ZDBF2"]
    cnv = ["CND1","CCNE1","CCNE124","CDKN2A","CNTN3","EGFR","ERBB2","FGFR1","FHIT","IRS2","KIF2A","KIT","KRAS","MDM2","MET","MYC","MYCL1","MYCN","NKX2-1","PTPRD","RASSF1","RB1","ROBO1","SOX2","SOX4","TP53"]

    dic = {"trans":trans,"mut":mut,"cnv":cnv}
    
    
    if args.gsva:
        
        genesets_df = gmt_to_dataframe(args.inputdir+'/c2.cp.kegg.v7.1.symbols.gmt')
        Train_df = gsva(Train[Train.columns[:-5]].T,genesets_df,method="gsva",tempdir=args.outdir).T
        Test_df = gsva(Test[Test.columns[:-5]].T,genesets_df,method="gsva",tempdir=args.outdir).T
        Train_df2 = pd.concat([Train_df,Train[Train.columns[-5:]]],axis=1)
        Test_df2 = pd.concat([Test_df,Test[Test.columns[-5:]]],axis=1)
        
        GSVA_model = AUC_Analysis(args.regular,Train_df2,Test_df2,Train_df2.columns[:-5],args.outdir,"GSVA")
        PCA_Analysis(GSVA_model.feature_names_in_,pd.concat([Train_df2,Test_df2]),"Label2",args.outdir,"GSVA")
        
        Model = open(args.outdir+'/LCNEC.pickle','wb')
        pickle.dump(GSVA_model,Model)
        Model.close()
        
    else:
        prior = []
        for i in args.prior.split(","):
            prior.extend(dic[i])
        
        Prior_model = AUC_Analysis(args.regular,Train,Test,prior,args.outdir,args.prior)
        PCA_Analysis(Prior_model.feature_names_in_,pd.concat([Train,Test]),"Label2",args.outdir,args.prior)
        
        Model = open(args.outdir+'/LCNEC.pickle','wb')
        pickle.dump(Prior_model,Model)
        Model.close()
        
if __name__ == "__main__":
    main()
