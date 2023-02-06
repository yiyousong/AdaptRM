import sys
import numpy as np
import torch
import utils as utils
import argparse
from sublayers import *
parser = argparse.ArgumentParser()
parser.add_argument('--model', default=1,help='Weak,Adapt,Encoder,Mixer')
parser.add_argument('--embedsize', default=1,help='4^__size__')
parser.add_argument('--dim', default=None)
parser.add_argument('--depth', default=None)
parser.add_argument('--drop', default=0.2)
parser.add_argument('--act', default='relu')
parser.add_argument('--out',default=None,help='37-40')
parser.add_argument('--data',default=None,help='37-40')
args, unknown = parser.parse_known_args()
model_list=['Weak','Adapt','Encoder','Mixer','Adapt2']
mer=int(args.embedsize)
drop=float(args.drop)
args.sep=False
kernelsize=7
adaptout=7
h2=32
# class Linout(nn.Module):
#     def __init__(self,in_size,out_size,hidden=2048,acti='relu',dropout=0.2):
#         super(Linout, self).__init__()
#         self.flat=nn.Flatten()
#         self.model = nn.Sequential(
#                 nn.Linear(in_size, hidden),
#                 # nn.BatchNorm1d(hidden),
#                 actfunc(acti),
#                 nn.Dropout(dropout),
#                 nn.Linear(hidden,h2),
#                 # nn.BatchNorm1d(hidden),
#                 actfunc(acti),
#                 nn.Dropout(dropout),
#                 nn.Linear(h2, out_size),nn.Sigmoid())
#     def forward(self,x):
#         x=self.flat(x)
#         out=self.model(x)
#         return out
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
src_vocab=4**mer
if args.data is None:
    targetlist = True
    target = np.arange(39, 43)
    savepath='/home/yiyou/model/zf_h32_%s_mer%d.model' % (model_list[int(args.model)], mer)
    sys.stdout = open('zf_h32_%s_mer%d.out' % (model_list[int(args.model)], mer), 'w')

#
else:
    targetlist = False
    target = int(args.data)
    savepath='/home/yiyou/model/zf_h512_model%s_mer%d_data%d.model' % (model_list[int(args.model)], mer, target)
    sys.stdout = open('zf_h512_model%s_mer%d_data%d.out' % (model_list[int(args.model)], mer, target), 'w')
if int(args.model) == 0:
    model = nn.Sequential(ReshapeTrim(),
                          Conv1dtranspose(in_chan=src_vocab, kernel_size=7, in_transpose=True, dropout=drop,
                                          acti=args.act, out_chan=64, pooling=True),
                          Conv1dtranspose(in_chan=64, kernel_size=5, out_chan=16, pooling=True, dropout=drop,
                                          acti=args.act),
                          AttentionMerge(out_size=100, in_size=6 * 16, acti=args.act))
elif int(args.model) == 1:
    model = nn.Sequential(Conv1dtranspose(in_chan=src_vocab, kernel_size=kernelsize, in_transpose=True, out_chan=32),
                          Conv1dtranspose(in_chan=32, kernel_size=kernelsize, out_chan=64, dropout=drop, acti=args.act),
                          nn.AdaptiveAvgPool1d(adaptout + kernelsize * 2 - 2),
                          Conv1dtranspose(in_chan=64, kernel_size=kernelsize, out_chan=32, dropout=drop, acti=args.act),
                          Conv1dtranspose(in_chan=32, kernel_size=kernelsize, out_chan=64, dropout=drop, acti=args.act),
                          Linout(in_size=adaptout * 64,hidden=512, acti=args.act, out_size=100)
                          )
elif int(args.model) == 2:
    model = Transformer(src_vocab=src_vocab, d_model=dim, dropout=drop, outsize=100, acti=args.act)
else:
    print('kernel_size=%d,adaptout=%d', (kernelsize, adaptout), flush=True)
    model = ConvMixer(vocab=src_vocab, sephead=args.sep, n_classes=100, dim=dim, depth=depth, dropout=drop,
                      kernel_size=kernelsize, adaptout=adaptout, ac1=args.act, ac2=args.act)
model = model.cuda()
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
optimizer = torch.optim.Adam(
    model.parameters(), lr=5e-5)
loss_func = torch.nn.functional.binary_cross_entropy
train_data_list=[]
train_label_list=[]
test_data_list=[]
test_label_list=[]
if targetlist:
    for targetint in target:
        data,label=utils.loadsingledata(targetint,mer=mer,shuffle=True,minlength=50,maxlength=500)
        train_data=data[:int(len(data)*0.8)]
        test_data=data[int(len(data)*0.8):]
        train_label = label[:int(len(label) * 0.8)]
        test_label = label[int(len(label) * 0.8):]
        print(len(data))
        train_data_list.append(train_data)
        train_label_list.append(train_label)
        test_data_list.append(test_data)
        test_label_list.append(test_label)

    for i in range(10):
            for k in range(4):
                train_data = train_data_list[k]
                train_label = train_label_list[k]
                test_data = test_data_list[k]
                test_label = test_label_list[k]
                train_auc, test_auc = utils.run_epoch(model, train_data, train_label,
                                                      test_data, test_label,
                                                      optimizer=optimizer, loss_func=loss_func)
                if i == 9:
                    print('final_test_auc=%f', test_auc[0])
else:
        data, label = utils.loadsingledata(target, mer=mer, shuffle=True, minlength=50, maxlength=500)
        train_data = data[:int(len(data) * 0.8)]
        test_data = data[int(len(data) * 0.8):]
        train_label = label[:int(len(label) * 0.8)]
        test_label = label[int(len(label) * 0.8):]
        print(len(data))
        for i in range(10):
                train_auc, test_auc = utils.run_epoch(model, train_data, train_label,
                                                      test_data, test_label,
                                                      optimizer=optimizer, loss_func=loss_func)
                if i == 9:
                    print('final_test_auc=%f', test_auc[0])
torch.save(model,savepath)