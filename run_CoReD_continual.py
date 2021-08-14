import torch.optim as optim
import argparse
from torchvision import transforms

import xception_origin
from Function_common import *
from Function_CoReD import *
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

#----------------------------
# Argument parser.
#----------------------------
parser = argparse.ArgumentParser(description='PyTorch CONTINUAL LEARNING')
# model
parser.add_argument('--name_sources', '-s', type=str, default='DeepFake', help='name of sources(more than one)(ex.DeepFake / DeepFake_Face2Face / DeepFake_Face2Face_FaceSwap)')
parser.add_argument('--name_target', '-t', type=str, default='Face2Face', help='name of target(only one)(ex.DeepFake / Face2Face / FaceSwap)')
parser.add_argument('--name_saved_folder1', '-folder1', type=str, default='CoReD', help='name of folder that will be made')
parser.add_argument('--name_saved_folder2', '-folder2', type=str, default='', help='name of folder that will be made in folder1 (just option)')
parser.add_argument('--path_data', '-d',type=str, default='./data/DeepFake', help='the folder of path must contains real/fake folders that is consists of images')
parser.add_argument('--path_preweight', '-w', '-path', type=str, default='./weights', help='the folder of path must source(s) folder that have model weights')

parser.add_argument('--lr', '-l', type=float, default=0.05, help='initial learning rate')
parser.add_argument('--KD_alpha', '-a', type=float, default=0.5, help='KD alpha')
parser.add_argument('--num_class', '-nc', type=int, default=2, help='number of classes')
parser.add_argument('--num_store', '-ns', type=int, default=5, help='number of stores')
parser.add_argument('--epochs', '-me', type=int, default=100, help='epochs')
parser.add_argument('--batch_size', '-nb', type=int, default=128, help='batch size')
parser.add_argument('--num_gpu', '-ng', type=str, default='2', help='excuted gpu number')

args = parser.parse_args()
set_seeds()

#hyperparameter
lr = args.lr
KD_alpha = args.KD_alpha
num_class = args.num_class
num_store_per = args.num_store
name_sources = args.name_sources
name_target = args.name_target
path_data = args.path_data
path_preweight = args.path_preweight

print('GPU num is' , args.num_gpu)
os.environ['CUDA_VISIBLE_DEVICES'] =str(args.num_gpu)

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
save_path = './{}_{}/{}/{}/'.format(name_sources,name_target,args.path_preweight,args.name_saved_folder2)
if '//' in save_path :
    save_path = save_path.replace('//','/')
try:
    if not os.path.isfile(save_path):
        os.makedirs(save_path)
except OSError:
    pass

print('name_source is ',name_source)
print('name_source2 is ',name_source2)
print('name_source3 is',name_source3)
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
if '_' not in name_sources: #Task1 (pre-train before continual learning)
    dicLoader,dicFReTAL = Make_DataLoader(path_data,name_source,name_target,train_aug=train_aug,val_aug=val_aug,mode_FReTAL=True,batch_size=args.batch_size)
else: #Task2-4 (continual learning)
    dicLoader,dicFReTAL = Make_DataLoader_continual(path_data,name_source=name_source,name_target=name_target,train_aug=train_aug,val_aug=val_aug,mode_FReTAL=True,batch_size=args.batch_size)

teacher_model, student_model = None,None
path_preweight = os.path.join(path_preweight,'{}'.format(name_sources))

print('-------prev_path_weight--------')
print(path_preweight)
print('-------------------------------')
teacher_model = xception_origin.xception(num_classes=2, pretrained='')
checkpoint =torch.load(path_preweight+'/model_best_accuracy.pth.tar')
teacher_model.load_state_dict(checkpoint['state_dict'])
teacher_model.eval(); teacher_model.cuda()

student_model = xception_origin.xception(num_classes=2,pretrained='')
checkpoint =torch.load(path_preweight+'/model_best_accuracy.pth.tar')
student_model.load_state_dict(checkpoint['state_dict'])
student_model.train(); student_model.cuda()

#FREASING THE TEACHER MODEL
teacher_model_weights = {}
for name, param in teacher_model.named_parameters():
    teacher_model_weights[name] = param.detach()        

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.1)
scaler = GradScaler()

list_correct = func_correct(teacher_model.cuda(),dicFReTAL['train_target_forCorrect'])
correct_loaders,list_ratio_loader = GetSplitLoaders_BinaryClasses(list_correct,dicFReTAL['train_target_dataset'],train_aug, num_store_per)
          
# FIXED THE AVG OF FEATURES. IT IS FROM A TEACHER MODEL
list_features = GetListTeacherFeatureFakeReal(teacher_model,correct_loaders)
list_features = np.array(list_features)
print(list_features[0].dtype)
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
        list_features_std = [[],[]]

        optimizer.zero_grad()
        with autocast(enabled=True):
            for j in range(num_store_per):
                for i in range(num_class):
                    feat = GetFeatureMaxpool(student_model,correct_loader_std[j][i])
                    if(list_features[i][j]==0):continue
                    feat = feat-torch.tensor(list_features[i][j]).cuda()
                    feat = torch.pow(feat.cuda(),2)
                    list_features_std[i].append(feat)
            teacher_outputs = teacher_model(inputs)
            outputs = student_model(inputs)
            loss_main = criterion(outputs, targets)
            loss_kd = loss_fn_kd(outputs,targets,teacher_outputs)
            sne_loss=0
            for fs in list_features_std:
                for ss in fs:
                    if ss.requires_grad:
                        sne_loss += ss
            loss = loss_main + loss_kd + sne_loss
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
        
    print("Epoch: {}/{} - CE_Loss: {:.4f} | KD_Loss: {:.4f} | OTHER_LOSS: {:.4f} | ACC: {:.4f}".format(epoch+1, epochs, np.mean(running_loss), np.mean(running_loss_kd),  np.mean(running_loss_other), correct / total))
    
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
    if (epoch+1)%20 ==0 or is_best_acc:
        if is_best_acc : best_acc = total_acc
        is_best_acc = True
        best_acc = max(correct / total,best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': student_model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()
        }, cnt=epoch, isAcc=is_best_acc,
            checkpoint=save_path,
            best_filename = '{}_epoch_{}.pth.tar'.format(args.path_preweight,epoch+1 if (epoch+1)%10==0 else ''))
        print('saved.........')