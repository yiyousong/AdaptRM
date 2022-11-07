import numpy as np
import torch
import sys
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import scripts as scrip
import argparse
header='/home/yiyou/model'
from sublayers import *
torch.set_num_threads(1)
np.random.seed(7)
parser = argparse.ArgumentParser()
parser.add_argument('--model', default=3,help='Weak,Adapt,Encoder,Mixer')
parser.add_argument('--embedsize', default=1,help='4^__size__')
parser.add_argument('--drop', default=0.2,help='dropout')
parser.add_argument('--act', default='relu')
parser.add_argument('--out',default=None,help='feedback output')

args, unknown = parser.parse_known_args()
num_epoch=20
src_vocab=4**int(args.embedsize)
model_list=['Weak','Adapt','Encoder','Mixer']
drop=float(args.drop)

if args.out is None:
    args.out='%s/Multi_%s_embedsize%d_dropout%.2f_dim%d_depth%d_act%s_sep_%s.out'%(header,model_list[int(args.model)],int(args.embedsize),drop,dim,depth,args.act,args.sep)
def loaddata(i):
    data, label = np2tensor(np.load("/data/yiyou/Multi_WeakRM/multi_shuffled/data_%d" % (i), allow_pickle=True),
                                np.load("/data/yiyou/Multi_WeakRM/multi_shuffled/label_%d" % (i), allow_pickle=True)
                            ,mer=int(args.embedsize))

    return data, label

if __name__ == '__main__':
    sys.stdout = open(args.out, 'w')
    if int(args.model) == 0:
        model = nn.Sequential(ReshapeTrim(),
                              Conv1dtranspose(in_chan=src_vocab, kernel_size=7, in_transpose=True,dropout=drop,acti=args.act, out_chan=64,pooling=True),
                              Conv1dtranspose(in_chan=64, kernel_size=5, out_chan=16,pooling=True,dropout=drop,acti=args.act),
                              AttentionMerge(out_size=100, in_size=6 * 16,acti=args.act))
    elif int(args.model) == 2:
        model=Transformer(src_vocab=src_vocab,d_model=256,dropout=drop,outsize=100,acti=args.act)
    elif int(args.model) == 3:
        kernel_size = int(args.kernelsize)
        adaptout = int(args.adaptout)
        print('kernel_size=%d,adaptout=%d', (kernel_size, adaptout), flush=True)
        model = ConvMixer(vocab=src_vocab, sephead=args.sep, n_classes=100, dim=256, depth=10, dropout=drop,
                          kernel_size=kernel_size, adaptout=adaptout, ac1=args.act, ac2=args.act)
    else:
        model = nn.Sequential(Conv1dtranspose(in_chan=src_vocab, kernel_size=7, in_transpose=True, out_chan=32),
                              Conv1dtranspose(in_chan=32, kernel_size=7, out_chan=64, dropout=drop, acti=args.act),
                              nn.AdaptiveAvgPool1d(19),
                              Conv1dtranspose(in_chan=64, kernel_size=7, out_chan=32, dropout=drop, acti=args.act),
                              Conv1dtranspose(in_chan=32, kernel_size=7, out_chan=64, dropout=drop, acti=args.act),
                              Linout(in_size=7 * 64, acti=args.act, out_size=100)
                              )
        model = model.cuda()
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-5)
    loss_func = torch.nn.functional.binary_cross_entropy
    tmp=np.arange(5)
    for k in range(num_epoch):
        for j in tmp:
            data_list=[]
            label_list=[]
            for i in range(4):
                data,label=loaddata(i+4*j)
                data_list.extend(data)
                label_list.extend(label)
            test_data,test_label=loaddata(20+j)
            scrip.run_epoch(model,data_list,label_list,test_data,test_label,optimizer=optimizer,loss_func=loss_func)
        tmp=np.flip(tmp)
    model_name='%s/Multi_%s_embedsize%d_dropout%.2f_dim%d_depth%d_act%s_sep_%s.model'%(header,model_list[int(args.model)],int(args.embedsize),drop,dim,depth,args.act,args.sep)
    torch.save(model,model_name )
    data_list = []
    label_list = []
    for i in range(5):
        data,label = loaddata(20 + i)
        data_list.extend(data)
        label_list.extend(label)
    print('total epoch%d'%(num_epoch))
    print('FINAL AUROC')
    auc=scrip.run_test_epoch(model, data_list, label_list)
    print('average auc: %.4f'%(np.average(auc[:,0])))
    sys.stdout.close()
