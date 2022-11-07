import numpy as np
import pandas as pd
from Bio import SeqIO
import argparse
import csv
import json
import torch
import torch.nn as nn
parser = argparse.ArgumentParser()
parser.add_argument('--specie', default="human",help='human,mouse,rat,zebrafish')
parser.add_argument('--fa',default='test.fa',help='Fasta File')
args, unknown = parser.parse_known_args()
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
                        , 'ac4c'
                        , 'hm5c'
                        , 'm7g'
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
                        ,'m7G_Ctrl'  ])
class Conv1dtranspose(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, dilation=1, padding=0, pooling=False,
                 in_transpose=False, out_transpose=False, groups=1, dropout=0.1):
        super(Conv1dtranspose, self).__init__()
        if padding == 'same':
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels=in_chan, out_channels=out_chan, padding=padding, groups=groups,
                              kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.in_transpose = in_transpose
        self.out_transpose = out_transpose
        # self.bn = nn.LayerNorm(out_chan)
        self.dropout = nn.Dropout(dropout)
        self.out=nn.ReLU()
        self.pooling = pooling
        if pooling:
            self.pool = nn.MaxPool1d(2)

    def forward(self, x, in_transpose=False):
        if in_transpose:
            x = torch.transpose(x, -1, -2)
        elif self.in_transpose:
            x = torch.transpose(x, -1, -2)
        x = self.conv(x)
        # x = self.bn(x)
        x = self.out(self.dropout(x))
        if self.pooling:
            x = self.pool(x)
        if self.out_transpose:
            x = torch.transpose(x, -1, -2)
        return x

class Linout(nn.Module):
    def __init__(self,in_size,out_size,hidden=2048,dropout=0.2,sephead=False):
        super(Linout, self).__init__()
        self.flat=nn.Flatten()
        if not sephead:
            self.model = nn.Sequential(
                nn.Linear(in_size, hidden),
                # nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear( hidden, out_size),nn.Sigmoid())
        else:
            hidden=hidden//16
            self.modelpart = nn.Sequential(
                nn.Linear(in_size, hidden),
                # nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_size), nn.Sigmoid())
            self.model=list2model(clones(self.modelpart,out_size))
    def forward(self,x):
        x=self.flat(x)
        out=self.model(x)
        return out
class list2model(nn.Module):
    def __init__(self, modellist):
        super(list2model, self).__init__()
        self.model=modellist
    def forward(self,x):
        out_list=[]
        for model in self.model:
            out=model(x)
            out_list.append(out)
        out=torch.cat(out_list)
        return out



def fasta2binonehot(data):
    # data is a list of sequence: [n,seqlength]
    # possibly need list version where seqlength differ
    data=np.squeeze(np.array(list(map(list, data))))
    A = np.zeros_like(data,dtype=int)
    C = np.zeros_like(data,dtype=int)
    G = np.zeros_like(data,dtype=int)
    U = np.zeros_like(data,dtype=int)
    A[data == 'A'] = 1
    C[data == 'C'] = 1
    G[data == 'G'] = 1
    U[data == 'U'] = 1
    U[data == 'T'] = 1
    A = A[..., np.newaxis]
    C = C[..., np.newaxis]
    G = G[..., np.newaxis]
    U = U[..., np.newaxis]
    bindata=np.append(A,C,axis=-1)
    bindata = np.append(bindata, G, axis=-1)
    bindata = np.append(bindata, U, axis=-1)
    return bindata
class AdaptRM(nn.Module):
    def __init__(self):
        super(AdaptRM, self).__init__()
        self.model = nn.Sequential(Conv1dtranspose(in_chan=4, kernel_size=7, in_transpose=True, out_chan=32),
                              Conv1dtranspose(in_chan=32, kernel_size=7, out_chan=64),
                              nn.AdaptiveAvgPool1d(19),
                              Conv1dtranspose(in_chan=64, kernel_size=7, out_chan=32),
                              Conv1dtranspose(in_chan=32, kernel_size=7, out_chan=64),
                              Linout(in_size=7 * 64, out_size=100))
    def forward(self, x):
        return self.model(x)
model=torch.load('Multi_Adapt.model',map_location=torch.device('cpu'))
specie=np.array(['human','mouse','rat','zebrafish'])
idx=np.where(specie==args.specie)
pred_list=[]
name_list=[]
for record in SeqIO.parse(args.fa, "fasta"):
    name_list.append(record.name)
    seq = fasta2binonehot(str(record.seq))
    seq=torch.as_tensor(seq[np.newaxis,:]).float()
    pred=model.forward(seq).view(-1)
    pred_list.append(pred.detach().numpy())
pred=np.asarray(pred_list)

idx=int(np.where(specie==args.specie)[0])
if idx==0:
    pred=pred[:,:25]
    out_list=out_list[:25]
elif idx==1:
    pred=pred[:,31:36]
    out_list=np.asarray(['mouse_brain','mouse_heart','mouse_kidney','mouse_liver','mouse_testis'])
elif idx==2:
    pred=pred[:,36:39]
    out_list=np.asarray(['rat_brain','rat_kidney','rat_liver'])
else:
    pred=pred[:,40:44]
    out_list=np.asarray(['$\mathregular{m^1}$A','$\mathregular{m^5}$C','$\mathregular{m^6}$A','$\mathregular{m^7}$G'])
predf=pd.DataFrame(pred, columns=out_list)
predf.insert(0, 'index', name_list)
predf.to_csv(pred.csv', index=False, header=True, sep=',')
