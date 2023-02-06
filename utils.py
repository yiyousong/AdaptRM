import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn import metrics

out_list = np.array(['Adrenal'
                        , 'Brainstem'
                        , 'Cerebellum'
                        , 'Cerebrum'
                        , 'Colon'
                        , 'EndoC'
                        , 'endometrial'
                        , 'Heart'
                        , 'HSC'
                        , 'Hypothalamus'
                        , 'islet'
                        , 'Kidney'
                        , 'Liver'
                        , 'lung'
                        , 'lymphocyte'
                        , 'Muscle'
                        , 'ovary'
                        , 'Prostate'
                        , 'Rectum'
                        , 'RWPE'
                        , 'Skin'
                        , 'Stomach'
                        , 'Testis'
                        , 'Thyroid'
                        , 'Urinary'
                        , 'h_b'
                        , 'h_k'
                        , 'h_l'
                        , 'm_b'
                        , 'm_h'
                        , 'm_k'
                        , 'm_l'
                        , 'm_t'
                        , 'r_b'
                        , 'r_k'
                        , 'r_l'
                        ,'m1A_Ctrl'
                        ,'m5C_Ctrl'
                        ,'m6A_Ctrl'
                        ,'m7G_Ctrl' ])
max_nclass=100

def run_train_epoch(model,data,labelsign,optimizer,loss_func,weights=None):
    pred_list=[]
    for i in range(len(data)):
        data1=data[i]
        label1=labelsign[i]
        pred=model(data1)
        optimizer.zero_grad()
        loss=signloss(loss_func,pred,label1,weights=weights)
        loss.backward()
        optimizer.step()
        pred_list.append(pred)
    result=evaluate(pred_list,labelsign,prefix='train')
    return result
def signloss(loss_func,pred,labelsign,weights=None):
        mask=labelsign**2
        mask[labelsign>0]=1
        if weights is not None:
            for i in range(len(weights)):
                mask[:,i]*=weights[i]
        label=(labelsign+1)//2
        loss=loss_func(pred,label,weight=mask)
        return loss
def evaluate(pred_list,labelsign=None,prefix=''):
    pred_list = torch.cat(pred_list).cpu().detach().numpy()
    label = torch.cat(labelsign).cpu().numpy()
    labelexist = np.sum(label ** 2, axis=0)
    index = np.arange(labelexist.shape[0])
    index = index[labelexist != 0]
    result=[]
    for i in index:
        y = label[:, i]
        pred = pred_list[:, i]
        pred = pred[y != 0]
        predint=(pred+0.5).astype(int)
        y = (y[y != 0] + 1) // 2
        y=y.astype(int)
        p=y[pred>0.5]
        n=y[pred<0.5]
        tp=np.sum(p)
        fn=np.sum(1-p)
        tn=np.sum(1-n)
        fp=np.sum(n)
        sense=tp/(fn+tp)
        speci=tn/(tn+fp)
        auc = metrics.roc_auc_score(y, pred)
        if np.sum(predint)*np.sum(1-predint)!=0:
            f1=metrics.f1_score(y,predint)
            mcc=metrics.matthews_corrcoef(y,predint)
        else:
            f1=mcc=0
        pr=metrics.average_precision_score(y,pred)
        result.append(np.array([auc,f1,mcc,pr,sense,speci]))
        # result.append(auc)
        print('%stask %d:\tauc: %.3f\tf1: %.3f\tmcc: %.3f\tap: %.3f\tsenseticity: %.3f\tspecificity: %.3f\t'% (prefix,i, auc,f1,mcc,pr,sense,speci), flush=True)
    result=np.array(result)
    return result
@torch.no_grad()
def run_test_epoch(model,data,labelsign,returnpred=False):
    model.eval()
    pred_list = []
    for i in range(len(data)):
        data1= data[i]
        pred=model(data1)
        pred_list.append(pred)
    result=evaluate(pred_list,labelsign,prefix='test ')
    model.train()
    if not returnpred:
        return result
    else:
        return result,pred_list
def run_epoch(model,traindata,trainlabelsign,testdata,testlabelsign,optimizer,loss_func,weights=None):
    train_auc=run_train_epoch(model, traindata, trainlabelsign, optimizer, loss_func,weights=weights)
    test_auc=run_test_epoch(model, testdata,testlabelsign)
    return train_auc,test_auc
def np2tensor(input_list, label_list=None, mer=1,newvar=0,minlength=0,maxlength=5000, chunk=False,instance_length=40,shuffle=False,train_ratio=0.8,posrep=1,negrep=1):
    input_tensor_list = []
    if label_list is not None:
        label_tensor_list = []
    index = np.arange(len(input_list))
    if shuffle:
        np.random.shuffle(index)
    for i in index:
        if label_list is not None:
            label = np.reshape(np.array(label_list[i]),[1,-1])
        input = np.squeeze(np.int32(input_list[i]))
        input = np.reshape(input, [-1, 4])
        if input.shape[0]>minlength:
            if input.shape[0]<maxlength:
                index1 = np.where(input == 1)[-1]
                if mer > 1:
                    idx = index1[mer - 1:]
                    for k in range(mer - 1):
                        idx = idx + (4 ** (mer - 1 - k)) * index1[k:k - mer + 1]
                    data = np.zeros([input.shape[0] - mer + 1, 4 ** mer + newvar])
                    data[np.where(input == 1)[0][:-mer + 1], idx] = 1
                else:
                    data=input
                if chunk:
                    data = data[:(len(data) // (instance_length - mer + 1)) * (instance_length - mer + 1)]
                    data = np.reshape(data, [-1, instance_length - mer + 1, 4 ** mer + newvar])
                else:
                    if mer >1:
                        data = np.reshape(data, [1, -1, 4 ** mer + newvar])
                    else:
                        data = np.reshape(input, [1, -1, 4])
                        if newvar > 0:
                            data = np.append(data, np.zeros([1, data.shape[1], newvar]), axis=-1)
                input = torch.as_tensor(data).float().cuda()
                input_tensor_list.append(input)
                if label_list is not None:
                    rep=0
                    if negrep > 1:
                        if np.sum(label) < 0:
                            rep = negrep
                    if posrep > 1:
                        if np.sum(label) > 0:
                            rep = posrep

                    label = torch.as_tensor(label).float().cuda()
                    label_tensor_list.append(label)
                    if rep>0:
                        for i in range(rep):
                            label_tensor_list.append(label)
                            input_tensor_list.append(input)

    if label_list is not None:
        return input_tensor_list, label_tensor_list
    else:
        return input_tensor_list
def loadsingledata(query,mer=1,newvar=0,expand_label=True,shuffle=False,minlength=0,maxlength=1000,posrep=1,negrep=1,train_ratio=0.8):
    if type(query) is str:
        if not query.isnumeric():
            query = np.where(out_list == query)[0][0]
        else:
            query = int(query)
    if query<25:
        data=np.load('/data/yiyou/Multi_WeakRM/%s_positive.pickle'%(out_list[query]),allow_pickle=True)
        neg_data=np.load('/data/yiyou/Multi_WeakRM/%s_negative.pickle'%(out_list[query]),allow_pickle=True)
        label=np.append(np.ones(len(data)),np.ones(len(neg_data))*-1)
        multi_label=np.zeros([len(label),max_nclass])
        multi_label[:,query]=label
        data.extend(neg_data)
        #sample using random.sample(dataset,n)
    elif query < 36:
        data = np.load('/data/yiyou/independent/sin%d.npy'%(query-28))
        label = np.load('/data/yiyou/independent/sin%d.label.npy'%(query-28))[:,query-28]
        multi_label = np.zeros([len(label), max_nclass])
        multi_label[:, query] = label
        # sample using random.sample(dataset,n)

    else:
        data = np.load('/data/yiyou/Multi_WeakRM/zebrafish/%s_positive.pickle' % (out_list[query]), allow_pickle=True)
        neg_data = np.load('/data/yiyou/Multi_WeakRM/zebrafish/%s_negative.pickle' % (out_list[query]), allow_pickle=True)
        label = np.append(np.ones(len(data)), np.ones(len(neg_data)) * -1)
        multi_label = np.zeros([len(label), max_nclass])
        multi_label[:, query] = label
        data.extend(neg_data)
    if not expand_label:
        multi_label=label
    datatensor,labeltensor=np2tensor(data,multi_label,mer=mer,newvar=newvar,shuffle=shuffle,minlength=minlength,maxlength=maxlength,posrep=posrep,negrep=negrep,train_ratio=train_ratio)
    return datatensor,labeltensor




