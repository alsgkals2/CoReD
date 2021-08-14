import sys
import os.path
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import transforms

import xception_origin
from EarlyStopping import EarlyStopping
from Function_common import *
from Function_FReTAL import *
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

#----------------------------
# Argument parser.
#----------------------------
parser = argparse.ArgumentParser(description='PyTorch CONTINUAL LEARNING')
# model
parser.add_argument('--lr', '-l', type=float, default=0.05, help='initial learning rate')
parser.add_argument('--KD_alpha', '-a', type=float, default=0.5, help='KD alpha')
parser.add_argument('--num_class', '-nc', type=int, default=2, help='number of classes')
parser.add_argument('--num_store_per', '-nsp', type=int, default=5, help='number of stores')
parser.add_argument('--epochs', '-e', type=int, default=100, help='epochs')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='batch size')
parser.add_argument('--num_gpu', '-ng', type=str, default='2', help='excuted gpu number')
parser.add_argument('--name_sources', '-s', type=str, default='DeepFake', help='name of sources(more than one)(ex.DeepFake / DeepFake_Face2Face / DeepFake_Face2Face_FaceSwap)')
parser.add_argument('--name_target', '-t', type=str, default='Face2Face', help='name of target(only one)(ex.DeepFake / Face2Face / FaceSwap)')
parser.add_argument('--name_saved_folder', '-nfolder', type=str, default='CoReD', help='name of folder that will be made')
parser.add_argument('--name_saved_folder2', '-nfolder2', type=str, default='', help='name of folder that will be made more specifically')

args = parser.parse_args()
random_seed = 2020
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


#hyperparameter
num_gpu = args.num_gpu
lr = args.lr
KD_alpha = args.KD_alpha
num_class = args.num_class
num_store_per = args.num_store_per
name_sources = args.name_sources
name_target = args.name_target
print('GPU num is' , num_gpu)
os.environ['CUDA_VISIBLE_DEVICES'] =str(num_gpu)

print('lr is ',lr)
print('KD_alpha is ',KD_alpha)
print('num_class is ',num_class)
print('num_store_per is ',num_store_per)

####################initialization########################
name_source, name_source2, name_source3 = name_sources,'',''
if '_' in name_sources:
    temp = name_sources.split('_')
    name_source = temp[0]
    name_source2 = temp[1]
    try:
        name_source3 = temp[2]
    except:
        print('name_source3 is empty')
save_path = './{}_{}/{}/{}/'.format(name_sources,name_target,args.name_saved_folder,args.name_saved_folder2)
print(f'save_path is {save_path}')
if '//' in save_path :
    save_path = save_path.replace('//','/')
try:
    if not os.path.isfile(save_path):
        os.makedirs(save_path)
except OSError:
    pass

print('name_source is ',name_source)
print('name_source2 is ',name_source2)
print('name_sourc_ ',name_source3)

print('name_target is ',name_target)
print('save_path is ',save_path)

#train & valid
train_aug = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

val_aug = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])
if '_' not in name_sources:
#     dicLoader,dicFReTAL = Make_DataLoader('/media/data1/sha/CLRNet_jpg25/CLRNet',name_source,name_target,train_aug=train_aug,val_aug=val_aug,mode_FReTAL=True)
    dicLoader,dicFReTAL = Make_DataLoader('/media/data1/sha/CLRNet',name_source,name_target,train_aug=train_aug,val_aug=val_aug,mode_FReTAL=True)

else:
#     dicLoader,dicFReTAL = Make_DataLoader_continual('/media/data1/sha/CLRNet_jpg25/CLRNet',name_source,name_source2,name_target,name_source3,train_aug=train_aug,val_aug=val_aug,mode_FReTAL=True)
    dicLoader,dicFReTAL = Make_DataLoader_continual('/media/data1/sha/CLRNet',name_source=name_source,name_target=name_target,train_aug=train_aug,val_aug=val_aug,mode_FReTAL=True)


teacher_model, student_model = None,None
# prev_path_weight = os.path.join('/home/mhkim/CoReD/FReTAL','{}/CONTINUAL_HQ'.format(name_sources, args.name_saved_folder))
# prev_path_weight = '/home/mhkim/CoReD/current_train_with_shadata/%s'%name_sources
# prev_path_weight = os.path.join(prev_path_weight,'FReTAL_HQ')
prev_path_weight = os.path.join('/home/mhkim/T-GD/sha_faceforensics_jpeg_comp100_xception','{}'.format(name_sources))

print('-------prev_path_weight--------')
print(prev_path_weight)
print('-------------------------------')
teacher_model = xception_origin.xception(num_classes=2, pretrained='')
checkpoint =torch.load(prev_path_weight+'/model_best_accuracy.pth.tar')
# checkpoint =torch.load(prev_path_weight+'/model_best_accuracy.pth.tar')
teacher_model.load_state_dict(checkpoint['state_dict'])
student_model = xception_origin.xception(num_classes=2,pretrained='')
checkpoint =torch.load(prev_path_weight+'/model_best_accuracy.pth.tar')
# checkpoint =torch.load(prev_path_weight+'/model_best_accuracy.pth.tar')
student_model.load_state_dict(checkpoint['state_dict'])
teacher_model.eval()
student_model.train()
teacher_model.cuda()
student_model.cuda()

#FREASING THE TEACHER MODEL
teacher_model_weights = {}
for name, param in teacher_model.named_parameters():
    teacher_model_weights[name] = param.detach()        

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.1)
scaler = GradScaler()

          
# FIXED THE AVG OF FEATURES. IT IS FROM A TEACHER MODEL
teacher_model, student_model = teacher_model.cuda(), student_model.cuda()

best_acc,epochs=0, args.epochs
print('epochs={}'.format(epochs))
is_best_acc = False
###########################################################
for epoch in range(epochs):
    running_loss = []
    running_loss_kd = []
    running_loss_other = []
    correct,total = 0,0
    teacher_model.eval()
    student_model.train()

    losses = AverageMeter()
    arc = AverageMeter()
    cls_losses = AverageMeter()
    sp_losses = AverageMeter()
    main_losses = AverageMeter()
    alpha = AverageMeter()
    real_acc = AverageMeter()
    fake_acc = AverageMeter()
    
    for batch_idx, (inputs, targets) in enumerate(dicLoader['train_target']):
        inputs, targets = inputs.cuda(), targets.cuda()
        sne_loss = None
        r = np.random.rand(1)
        if r > 0.8:
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            tt = targets[rand_index]
            boolean = targets != tt #THIS IS ALWAYS ATTACHING THE OPPOSITED THE 'SMALL PIECE OF A DATA'
            if True in boolean:
                rand_index = rand_index[boolean]
                lam = np.random.beta(0.5,0.5)
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[boolean, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

        correct_loader_std,_ = correct_binary(student_model.cuda(), inputs, targets)
#         list_features_std = [[] for i in range(num_class)]
        list_features_std = [[],[]]

        optimizer.zero_grad()
        with autocast(enabled=True):
            teacher_outputs = teacher_model(inputs)
            outputs = student_model(inputs)
            loss_main = criterion(outputs, targets)
            loss_kd = loss_fn_kd(outputs,targets,teacher_outputs)
            loss = loss_main + loss_kd
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += len(targets)
        running_loss.append(loss_main.cpu().detach().numpy())
        running_loss_kd.append(loss_kd.cpu().detach().numpy())
        try:
            running_loss_other.append(sne_loss.cpu().detach().numpy())
        except AttributeError:
            pass
        
    print("Epoch: {}/{} - CE_Loss: {:.4f} | KD_Loss: {:.4f} | ACC: {:.4f}".format(epoch+1, epochs, np.mean(running_loss), np.mean(running_loss_kd), correct / total))
    
    #validataion
    test_loss, test_auroc, test_acc = Test(dicLoader['val_target'], student_model, criterion)
    total_acc = test_acc
    source_loss, source_auroc, source_acc = Test(dicLoader['val_source'], student_model, criterion)
    total_acc += source_acc
    if name_source2:
        source_loss2, source_auroc2, source_acc2 = Test(dicLoader['val_source2'], student_model, criterion)
        total_acc += source_acc2
    if name_source3:
        source_loss3, source_auroc3, source_acc3 = Test(dicLoader['val_source3'], student_model, criterion)
        total_acc += source_acc3
        
    is_best_acc = total_acc > best_acc  
#     best_acc = max(total_acc, best_acc)
    if (epoch+1)%20 ==0 or is_best_acc:
        if is_best_acc : best_acc = total_acc
        is_best_acc = True
        best_acc = max(correct / total,best_acc)
        save_checkpoint_for_unlearning({
            'epoch': epoch + 1,
            'state_dict': student_model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()
        }, cnt=epoch, isAcc=is_best_acc,
            checkpoint=save_path,
            best_filename = '{}_epoch_{}.pth.tar'.format(args.name_saved_folder,epoch+1 if (epoch+1)%10==0 else ''))
        print('saved.........')