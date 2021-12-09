import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import functional as F
from Function_common import CustumDataset
def _GetIndex(data_1):
    idx = -1
    if data_1 > 0.5 and data_1 <= 0.6:
        idx = 0
    elif data_1 >0.6 and data_1 <= 0.7:
        idx = 1
    elif data_1 >0.7 and data_1 <= 0.8:
        idx = 2
    elif data_1 >0.8 and data_1 <= 0.9:
        idx = 3
    elif data_1 >0.9 and data_1 <= 1.0:
        idx = 4
    return idx

def _GetIndex_avgfeat(data_1):
    idx = -1
    if data_1 > 0.5 and data_1 <= 0.6:
        idx = 0
    return idx
        
def GetSplitLoaders_BinaryClasses(list_correct,dataset,train_aug=None,num_store_per=5,batch_size=128):
    correct_loader=[[],[]]
    for i in range(num_store_per):
        list_temp = [list_correct[i][0],list_correct[i][1]]
        for rf in range(len(list_temp)):
            if not list_temp[rf] :
                correct_loader[rf].append([])
                continue
            custum = CustumDataset(np.array(dataset.data[list_correct[i][rf]]),
                                   np.array(dataset.target[list_correct[i][rf]]),
                                   train_aug)
            correct_loader[rf].append(DataLoader(custum,
                                     batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))
    
    list_length_realfakeloader = [[len(j.dataset) if j else 0 for j in i] for i in correct_loader]
    print("list_length_realfakeloader :", list_length_realfakeloader)
    return correct_loader,np.array(list_length_realfakeloader)/len(dataset.target)

def GetSplitLoadersRealFake(list_correct,dataset,train_aug=None,num_store_per=5):
    correct_loader=[[],[]]
    for i in range(num_store_per):
        list_temp = [list_correct[i][0],list_correct[i][1]]
        for rf in range(len(list_temp)):
            if not list_temp[rf] :
                correct_loader[rf].append([])
                continue
            temp_dataset = copy.deepcopy(dataset)
            temp_dataset.data = np.array(temp_dataset.data[list_correct[i][rf]])
            temp_dataset.target = np.array(temp_dataset.target[list_correct[i][rf]])
            custum = CustumDataset(temp_dataset.data,temp_dataset.target,train_aug)
            correct_loader[rf].append(DataLoader(custum,
                                     batch_size=200, shuffle=False, num_workers=4, pin_memory=True))    
    
    list_length_realfakeloader = [[len(j.dataset) if j else 0 for j in i] for i in correct_loader]
    return correct_loader,np.array(list_length_realfakeloader)/len(dataset.target),save_ceckpoint_for_unlearning

def GetListTeacherFeatureFakeReal(model, loader,mode='X',showScatter = False):
    
    list_features = [[],[]]
    maxpool = nn.MaxPool2d(4)
    model.eval()
    with torch.no_grad():
        train_results, labels = [[],[]],[[],[]]
        for i in range(len(loader)):
            for j in range(len(loader[i])):
                if not loader[i][j] :
                    train_results[i].append([])
                    list_features[i].append(torch.tensor(0))
                    continue
                temp = None
                for _,(img, label) in enumerate(loader[i][j]):
                    train_results[i].append(model(img.cuda()).cpu().detach().numpy())
                    labels[i].append(label)
                    if mode == 'E':
                        test = model.extract_features(img.cuda())
                    else:
                        test = model.features(img.cuda())
                    if temp is not None:
                        temp = torch.cat((maxpool(test),temp))
                    else:
                        temp = maxpool(test)
                temp = torch.mean(temp,dim=1)
                temp = torch.mean(temp,dim=0)        
                list_features[i].append(temp.detach().cpu())
                if showScatter:
                    train_results[i] = np.concatenate(train_results[j])
                    labels = np.concatenate(labels[j])    
                    plt.figure(figsize=(5, 5), facecolor="azure")
                    for label in np.unique(labels[j]):
                        tmp = train_results[i][labels[j]==label]
                        plt.scatter(tmp[:, 0], tmp[:, 1], label=label)
                else: continue
                plt.legend()
                plt.show()
    return list_features

def func_correct(model, data_loader):
    list_correct = [[[],[]] for i in range(5)]
    model.eval()
    cnt=0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            _inputs = inputs.cuda()
            _targets = targets.cuda()
            outputs = model(_inputs)
            temp = F.softmax(outputs,dim=1)
            for l in range(len(_targets)):
                idx = _GetIndex(temp[l][_targets[l]].data)
                if idx >= 0:
                    if _targets[l]==0 : 
                        list_correct[idx][0].append(cnt)
                    else : list_correct[idx][1].append(cnt)
                cnt += 1
        return list_correct

def func_correct_avgfeat(model, data_loader):
    list_correct = [[[],[]] for i in range(5)]
    model.eval()
    cnt=0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            _inputs = inputs.cuda()
            _targets = targets.cuda()
            outputs = model(_inputs)
            temp = F.softmax(outputs,dim=1)
            for l in range(len(_targets)):
                idx = _GetIndex_avgfeat(temp[l][_targets[l]].data)

                if idx >= 0:
                    if _targets[l]==0 : 
                        list_correct[idx][0].append(cnt)
                    else : list_correct[idx][1].append(cnt)
                cnt+=1
        return list_correct

def GetRatioData(list_real_fake,correct_cnt):
    if correct_cnt == 0 :return 0
    list_length_realfakeloader = np.array([[len(j) if j else 0 for j in i] for i in list_real_fake])
    return list_length_realfakeloader/correct_cnt

def correct_binary(model, inputs, targets, b_ratio_Data = False):
    list_correct = [[[], []] for i in range(5)]
    model.eval()
    cnt = 0
    correct_cnt=0
    ratio_data = None
    with torch.no_grad():
        _inputs = inputs.cuda()
        _targets = targets.cuda()
        outputs = model(_inputs)
        temp = nn.Softmax(dim=1)(outputs)
        temp = temp.cpu()
        for l in range(len(_targets)):
            idx = _GetIndex(temp[l][_targets[l]].data)
            if idx >= 0:
                correct_cnt+=1
                if _targets[l] == 0:
                    list_correct[idx][0].append((cnt,_inputs[l]))
                else:
                    list_correct[idx][1].append((cnt,_inputs[l]))
            cnt += 1
        if b_ratio_Data :
            ratio_data = GetRatioData(list_correct,correct_cnt)
    return list_correct, ratio_data

def correct_2(model, inputs, targets):
    list_correct = [[[], []] for i in range(5)]
    model.eval()
    cnt = 0
   
    _inputs = inputs.cuda()
    _targets = targets.cuda()
    with torch.no_grad():
        outputs = model(_inputs)
        temp = nn.Softmax(dim=1)(outputs)
        for l in range(len(_targets)):
            idx = _GetIndex_2(temp[l][_targets[l]].data)
            if idx >= 0:
                correct_cnt+=1
                if _targets[l] == 0:
                    list_correct[idx][0].append((cnt,_inputs[l]))
                else:
                    list_correct[idx][1].append((cnt,_inputs[l]))
            cnt += 1
    return list_correct

def correct_binary_avgfeat(model, inputs, targets, b_ratio_Data = False):
    list_correct = [[[], []] for i in range(5)]
    model.eval()
    cnt = 0
    correct_cnt=0
    ratio_data = None
    with torch.no_grad():
        _inputs = inputs.cuda()
        _targets = targets.cuda()
        outputs = model(_inputs)
        temp = nn.Softmax(dim=1)(outputs)
        for l in range(len(_targets)):
            idx = _GetIndex_avgfeat(temp[l][_targets[l]].data)
            if idx >= 0:
                correct_cnt+=1
                if _targets[l] == 0:
                    list_correct[idx][0].append((cnt,_inputs[l]))
                else:
                    list_correct[idx][1].append((cnt,_inputs[l]))
            cnt += 1
        if b_ratio_Data :
            ratio_data = GetRatioData(list_correct,correct_cnt)
    return list_correct, ratio_data

def correct_2_avgfeat(model, inputs, targets):
    list_correct = [[[], []] for i in range(5)]
    model.eval()
    cnt = 0
    _inputs = inputs.cuda()
    _targets = targets.cuda()
    with torch.no_grad():
        for l in range(len(_targets)):
            idx = 0
            if idx >= 0:
                correct_cnt+=1
                if _targets[l] == 0:
                    list_correct[idx][0].append((cnt, _inputs[l]))
                else:
                    list_correct[idx][1].append((cnt, _inputs[l]))
            cnt += 1
    return list_correct


def GetFeatureMaxpool(model,list_loader,mode='X'): #list_loader : consists of index,data
    feat = None
    maxpool = nn.MaxPool2d(4) #If using other networks, we can consider the number '4'
    if not list_loader : return 0
    for idx, img in list_loader:
        img = torch.reshape(img,(1,3,128,128))
        if mode == 'E':
            feat_std = model.extract_features(img.cuda())
        else:
            feat_std = model.features(img.cuda())
        feat_std = feat_std.cuda()
        if feat is not None:
            feat = torch.cat((maxpool(feat_std),feat))
        else:
            feat = maxpool(feat_std)
    if feat is None :
        feat = torch.tensor(0)
    else:
        feat = torch.mean(feat, dim=1)
        feat = torch.mean(feat, dim=0)
    return feat.view(1,-1)