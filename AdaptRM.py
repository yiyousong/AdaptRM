import numpy as np
import torch
import sys
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import torch_scripts_sep as scrip
import argparse
from sublayers import *
torch.set_num_threads(1)
np.random.seed(7)
parser = argparse.ArgumentParser()
parser.add_argument('--model', default=3,help='0,1,2,3 for Weak,Adapt,Encoder,Mixer')
parser.add_argument('--embedsize', default=1,help='4^__size__')
parser.add_argument('--dim', default=None)
parser.add_argument('--depth', default=None)
parser.add_argument('--header', default='/home/yiyou/')
parser.add_argument('--datadir', default='/data/yiyou/')
parser.add_argument('--drop', default=0.2)
parser.add_argument('--act', default='relu')
parser.add_argument('--out',default=None,help='feedback output')
parser.add_argument('--sep',default=False,help='sep head')
parser.add_argument('--weightloss',default=False,help='')
args, unknown = parser.parse_known_args()
num_epoch=20
args.header='/home/yiyou/'
args.datadir='/data/yiyou/'
src_vocab=4**int(args.embedsize)
model_list=['Weak','Adapt','Encoder','Mixer']
drop=float(args.drop)
if args.dim is None:
    if int(args.model) == 2:
        dim=256
    elif int(args.model)==3:
        dim=256
    else:
        dim=64
else:
    dim=int(args.dim)
if args.depth is None:
    if int(args.model) == 2:
        depth=5
    elif int(args.model)==3:
        depth=10
    else:
        depth=0
else:
    depth=int(args.depth)
if args.out is None:
    args.out='%sout/Multi_%s_embedsize%d_dropout%.2f_dim%d_depth%d_act%s_sep_%s_weightloss_%s.out'%(args.header,model_list[int(args.model)],int(args.embedsize),drop,dim,depth,args.act,args.sep,args.weightloss)
lossweight=np.array([0.79377679, 0.55017606, 3.14465409, 1.33832976, 0.67925554,
       0.53384583, 3.17662008, 2.08594076, 0.9046499 , 0.53648069,
       1.49120191, 2.54452926, 0.68709633, 1.56838143, 0.77006006,
       3.47705146, 2.06526229, 0.68898994, 1.56543519, 0.86625087,
       2.51382604, 2.13766567, 0.76593137, 1.81950509, 3.20512821])


def loaddata(i):
    data, label = np2tensor(np.load("%sMulti_WeakRM/multi_shuffled/data_%d" % (args.datadir,i), allow_pickle=True),
                                np.load("%sMulti_WeakRM/multi_shuffled/label_%d" % (args.datadir,i), allow_pickle=True)
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
        model=Transformer(src_vocab=src_vocab,d_model=dim,dropout=drop,outsize=100,acti=args.act)
    elif int(args.model) == 1:
        model = nn.Sequential(Conv1dtranspose(in_chan=src_vocab, kernel_size=7, in_transpose=True, out_chan=32),
                              Conv1dtranspose(in_chan=32, kernel_size=7, out_chan=64, dropout=drop, acti=args.act),
                              nn.AdaptiveAvgPool1d(19),
                              Conv1dtranspose(in_chan=64, kernel_size=7, out_chan=32, dropout=drop, acti=args.act),
                              Conv1dtranspose(in_chan=32, kernel_size=7, out_chan=64, dropout=drop, acti=args.act),
                              Linout(in_size=7 * 64, acti=args.act, out_size=100, sephead=args.sep)
                              )
    else:
        model=ConvMixer(vocab=src_vocab,sephead=args.sep, n_classes=100,dim=dim,depth=depth,dropout=drop,kernel_size=7,adaptout=5,ac1=args.act,ac2=args.act)
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
            if args.weightloss:
                scrip.run_epoch(model, data_list, label_list, test_data, test_label, optimizer=optimizer,
                                loss_func=loss_func,weights=lossweight)
            else:
                scrip.run_epoch(model,data_list,label_list,test_data,test_label,optimizer=optimizer,loss_func=loss_func)
        tmp=np.flip(tmp)
    model_name='%s/model/Multi_%s_embedsize%d_dropout%.2f_dim%d_depth%d_act%s_sep_%s_weightloss_%s.model'%(args.header,model_list[int(args.model)],int(args.embedsize),drop,dim,depth,args.act,args.sep,args.weightloss)
    torch.save(model,model_name )
    data_list = []
    label_list = []
    for i in range(5):
        data,label = loaddata(20 + i)
        data_list.extend(data)
        label_list.extend(label)
    print('total epoch%d'%(num_epoch))
    print('FINAL AUROC')
    auc,predlist=scrip.run_test_epoch(model, data_list, label_list,returnpred=True)
    print('average auc: %.4f'%(np.average(auc[:,0])))
    np.save('%sout/Multi_%s_embedsize%d_dropout%.2f_dim%d_depth%d_act%s_sep_%s_weightloss_%s_pred.npy' % (
        args.header,model_list[int(args.model)], int(args.embedsize), drop, dim, depth, args.act, args.sep,args.weightloss),predlist)
    np.save('%sout/Multi_%s_embedsize%d_dropout%.2f_dim%d_depth%d_act%s_sep_%s_weightloss_%s_label.npy' % (
        args.header,model_list[int(args.model)], int(args.embedsize), drop, dim, depth, args.act, args.sep,args.weightloss),np.asarray(label_list))

    sys.stdout.close()