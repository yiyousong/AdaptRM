import pandas as pd
import numpy as np
import pickle
train_data_list = []
train_label_list = []
out_list=np.array(['Adrenal'
        ,'Brainstem'
        ,'Cerebellum'
        ,'Cerebrum'
        ,'Colon'
        ,'EndoC'
        ,'endometrial'
        ,'Heart'
        ,'HSC'
        ,'Hypothalamus'
        ,'islet'
        ,'Kidney'
        ,'Liver'
        ,'lung'
        ,'lymphocyte'
        ,'Muscle'
        ,'ovary'
        ,'Prostate'
        ,'Rectum'
        ,'RWPE'
        ,'Skin'
        ,'Stomach'
        ,'Testis'
        ,'Thyroid'
        ,'Urinary'])
multisize=25
for i in range(multisize):
    neg_data=np.load('/data/yiyou/Multi_WeakRM/%s_negative.pickle'%(out_list[i]), allow_pickle=True)
    pos_data=np.load('/data/yiyou/Multi_WeakRM/%s_positive.pickle'%(out_list[i]), allow_pickle=True)
    data=np.append(pos_data,neg_data,axis=0)
    label=np.zeros([len(neg_data)+len(pos_data),100])
    label[:len(pos_data),i] = 1
    label[len(pos_data):,i]=-1
    index=np.arange(len(label))
    np.random.shuffle(index)
    train_label=label[index]
    train_data=data[index]
    if i>0:
        train_data_list =np.append(train_data_list,train_data)
        train_label_list = np.append(train_label_list,train_label,axis=0)
    else:

        train_data_list =train_data
        train_label_list =train_label
    assert len(train_label)==len(train_data)
print(len(train_label_list))
print(len(train_data_list))
tindex = np.arange(len(train_label_list))
np.random.shuffle(tindex)
train_data_list=train_data_list[tindex]
train_label_list=train_label_list[tindex]
for i in range(25):
    with open('/data/yiyou/Multi_WeakRM/multi_shuffled/data_%d'%(i),'wb+') as f:
        pickle.dump(train_data_list[tindex[int(len(train_label_list)*(i)/25):int(len(train_label_list)*(i+1)/25)]],f)
    with open('/data/yiyou/Multi_WeakRM/multi_shuffled/label_%d' % (i), 'wb+') as f:
        pickle.dump(train_label_list[
                        tindex[int(len(train_label_list) * (i) / 25):int(len(train_label_list) * (i + 1) / 25)]], f)