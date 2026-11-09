from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,average_precision_score,precision_recall_curve,auc,brier_score_loss,confusion_matrix,ConfusionMatrixDisplay)
from core import probability

def measure(y,p,t):
    pred=(p>=t).astype(int)
    prec,rec,_=precision_recall_curve(y,p)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(Accuracy=accuracy_score(y,pred),Precision=precision_score(y,pred,zero_division=0),Recall=recall_score(y,pred,zero_division=0),F1=f1_score(y,pred,zero_division=0),AP=average_precision_score(y,p),PR_AUC_trapezoid=auc(rec,prec),Brier=brier_score_loss(y,p),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))

def build_reports(bundle,splits,destination):
    dest=Path(destination); dest.mkdir(exist_ok=True,parents=True)
    rows=[]; fig,ax=plt.subplots(figsize=(7,5)); predictions={}
    for name,(x,y) in splits.items():
        p=probability(bundle,x); predictions[name]=(y,p)
        rows.append({'Split':name,'Threshold':bundle['threshold'],**measure(y,p,bundle['threshold'])})
        if name!='Train':
            prec,rec,_=precision_recall_curve(y,p)
            ax.plot(rec,prec,label=f'{name} AP={average_precision_score(y,p):.3f}')
            ax.axhline(np.mean(y),linestyle=':',alpha=.4)
    ax.set(xlabel='Recall',ylabel='Precision',title='Precision–Recall (horizontal lines: prevalence)'); ax.legend()
    fig.tight_layout();fig.savefig(dest/'pr_curve.png',dpi=160);plt.close(fig)
    table=pd.DataFrame(rows);table.to_csv(dest/'metrics.csv',index=False)
    if {'Train','Validation'}<=set(table.Split):
        fig,ax=plt.subplots(figsize=(7,4))
        table.set_index('Split').loc[['Train','Validation'],['Precision','Recall','F1','AP']].T.plot.bar(ax=ax,ylim=(0,1))
        fig.tight_layout();fig.savefig(dest/'overfit.png',dpi=160);plt.close(fig)
    if 'Test' in predictions:
        y,p=predictions['Test']; fig,ax=plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay.from_predictions(y,p>=bundle['threshold'],labels=[0,1],ax=ax,colorbar=False)
        fig.tight_layout();fig.savefig(dest/'confusion_test.png',dpi=160);plt.close(fig)
    for name,(x,y) in splits.items():
        if name not in predictions:continue
        pd.DataFrame({'source_row':x.index,'actual':np.asarray(y),'probability':predictions[name][1]}).to_csv(dest/f'predictions_{name.lower()}.csv',index=False)
    return table
