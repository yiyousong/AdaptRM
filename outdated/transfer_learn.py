import numpy as np
import torch
import math
import sys
import utils as utils
import argparse
from sublayers import *
model_path='/home/yiyou/may/model/Multi_Adapt2_embedsize1_dropout0.20_dim64_depth0_actrelu_sep_False_kernel_size7_adaptout7.model'
parser = argparse.ArgumentParser()
parser.add_argument('--retain',default=False,help='T for no submodel')
parser.add_argument('--model',default=model_path,help='model_path')
parser.add_argument('--target', default=None,help='sep by comma')
parser.add_argument('--out_list',default=None,help="'h_b','h_k','h_l','m_b','m_h','m_k','m_l','m_t','r_b','r_k','r_l'" )
parser.add_argument('--independent', default=1,help='1 for independent')
args, unknown = parser.parse_known_args()
mer=1
model_path=args.model
model_name=model_path.split('_')[2]
sys.stdout= open('transfer_41_%s.out'%(model_name),'w+')

out_list=np.array(['h_b'
        ,'h_k'
        ,'h_l'
        ,'m_b'
        ,'m_h'
        ,'m_k'
        ,'m_l'
        ,'m_t'
        ,'r_b'
        ,'r_k'
        ,'r_l'])
if args.target is None:
    targetint = np.arange(25, 36)
else:
    target=args.target.split(',')
    targetint=[]
    for query in target:
        if not query.isnumeric():
            query = np.where(out_list == query)[0]
        else:
            query=int(query)
        targetint.append(query+25)

model1=torch.load(args.model)
if not args.retain:
    input_size = model1.model[0].conv.in_channels
    conv_output_size = model1.model[-3][-1].conv.out_channels
    out_feature=model1.model[-1].model[-2].out_features
    model1.model=model1.model[:-1]
    mer=int(math.log(input_size,4))
    modelhead=Linout(12*conv_output_size,out_size=out_feature).cuda()
    submodel=nn.Sequential(
        Conv1dtranspose(in_transpose=True,in_chan=input_size,out_chan=64,kernel_size=7,pooling=True),
        Conv1dtranspose(in_chan=64,out_chan=conv_output_size,kernel_size=7,pooling=True),
        # nn.Flatten()
        #5*64
    ).cuda()
    model1=model1.cuda()
    model=Mergemodel(modelhead,[model1,submodel]).cuda()
    optimizer = torch.optim.Adam(
        # model.parameters(),
        [{'params': model.modelhead.parameters()},
                    {'params': model.modellist[-1].parameters(),'lr':5e-5}],
        lr=1e-5)
else:
    model = model1.cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-5)
loss_func = torch.nn.functional.binary_cross_entropy
datanestedlist=[]
labelnestedlist=[]
testdatanestedlist=[]
testlabelnestedlist=[]
for i in targetint:
    data,label=utils.loadsingledata(i,mer=mer,shuffle=True)
    train_data=data[:int(len(data)*0.8)]
    test_data=data[int(len(data)*0.8):]
    train_label = label[:int(len(label) * 0.8)]
    test_label = label[int(len(label) * 0.8):]
    datanestedlist.append(train_data)
    labelnestedlist.append(train_label)
    testdatanestedlist.append(test_data)
    testlabelnestedlist.append(test_label)
print('independent')
data_list=[]
label_list=[]
for i in np.arange(28,39):
    data = np.load('/data/yiyou/Multi_WeakRM/independent/%s.pickle' % (out_list[i]), allow_pickle=True)
    label = np.load('/data/yiyou/Multi_WeakRM/independent/%s.label.pickle' % (out_list[i]), allow_pickle=True)
    data_tensor,label_tensor=utils.np2tensor(data,label,mer=mer)
    data_list.append(data_tensor)
    label_list.append(label_tensor)
for i in range(10):
    test_auc_list = []
    test_perf_list=[]
    for j in range(len(targetint)):
        train_auc=utils.run_train_epoch(model,datanestedlist[j],labelnestedlist[j],
                        optimizer=optimizer,loss_func=loss_func)
    for j in range(len(targetint)):
        test_auc=utils.run_test_epoch(model,testdatanestedlist[j], testlabelnestedlist[j])
        test_auc_list.append(np.squeeze(test_auc))
    test_auc_list = np.array(test_auc_list)
    print('average test auc: %f'%(np.average(test_auc_list[:,0])),flush=True)
    for k in range(11):
        testperformance=utils.run_test_epoch(model,data_list[k],label_list[k])
        test_perf_list.append(np.squeeze(testperformance))
    test_perf_list=np.array(test_perf_list)
    print('average independent test auc: %f'%(np.average(test_perf_list[:,0])),flush=True)
np.save('%s_fine_tune_41bp_performance.npy'%(model_name),test_auc_list)
np.save('transfer_41_adapt_size%d.npy' % (mer), testperformance)
torch.save(model,args.model.split('.model')[0]+'_fine_tune_41bp.model')
# print('______________train__on__all___datasets________',flush=True)
# for i in np.arange(25):
#     data,label=utils.loadsingledata(i)
#     train_data=data[:int(len(data)*0.8)]
#     test_data=data[int(len(data)*0.8):]
#     train_label = label[:int(len(label) * 0.8)]
#     test_label = label[int(len(label) * 0.8):]
#     datanestedlist.append(train_data)
#     labelnestedlist.append(train_label)
#     testdatanestedlist.append(test_data)
#     testlabelnestedlist.append(test_label)
# for i in range(10):
#     print('______________epoch___%d_____'%(i), flush=True)
#     for j in range(len(targetint)+25):
#         utils.run_epoch(model,datanestedlist[j],labelnestedlist[j],
#                         testdatanestedlist[j],testlabelnestedlist[j],
#                         optimizer=optimizer,loss_func=loss_func)
#     for j in range(len(targetint)):
#         utils.run_epoch(model,datanestedlist[j],labelnestedlist[j],
#                         testdatanestedlist[j],testlabelnestedlist[j],
#                         optimizer=optimizer,loss_func=loss_func)
#
# torch.save(model,args.model+'fine_tune2')
