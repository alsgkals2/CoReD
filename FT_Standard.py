from Function_common import *
from Function_CoReD import *
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.optim as optim

def Train(args):
    lr = args.lr
    KD_alpha = args.KD_alpha
    num_class = args.num_class
    num_store_per = args.num_store

    print('GPU num is' , args.num_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] =str(args.num_gpu)
    savepath = f'./{args.name_sources}_{args.name_target}/{args.name_folder2}/'
    if '//' in savepath :
        savepath = savepath.replace('//','/')
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    print(f'save path : {savepath}')
    print('lr is ',lr)
    print('KD_alpha is ',KD_alpha)
    print('num_class is ',num_class)
    print('num_store_per is ',num_store_per)


    dicLoader,dicCoReD, dicSourceName = initialization(args)
    teacher_model, student_model = load_models(args.weigiht, nameNet='Xception', num_gpu=args.num_gpu)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.1)
    scaler = GradScaler()
    _list_correct = func_correct(teacher_model.cuda(),dicCoReD['train_target_forCorrect'])
    train_aug, _ = get_augs()
    _correct_loaders, _ = GetSplitLoaders_BinaryClasses(_list_correct, dicCoReD['train_target_dataset'], train_aug, num_store_per)
            
    # FIXED THE AVG OF FEATURES. IT IS FROM A TEACHER MODEL
    list_features = GetListTeacherFeatureFakeReal(teacher_model,_correct_loaders)
    list_features = np.array(list_features)

    best_acc,epochs=0, args.epochs
    print('epochs={}'.format(epochs))
    is_best_acc = False
    
    for epoch in range(epochs):
        running_loss = []
        running_loss_other = []
        correct,total = 0,0
        teacher_model.eval()
        student_model.train()
        losses = AverageMeter()
        main_losses = AverageMeter()
        for(inputs, targets) in tqdm(dicLoader['train_target']):
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

            optimizer.zero_grad()
            with autocast(enabled=True):     
                teacher_outputs = teacher_model(inputs)
                teacher_loss = criterion(teacher_outputs, targets)
                outputs = student_model(inputs)
                loss_main = criterion(outputs, targets)
                loss =  loss_main
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.update(loss.data.tolist(), inputs.size(0))
            main_losses.update(loss_main.tolist(), inputs.size(0))
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += len(targets)
            running_loss.append(loss_main.cpu().detach().numpy())
        
        print("Epoch: {}/{} - CE_Loss: {:.4f} | ACC: {:.4f}".format(epoch+1, epochs, np.mean(running_loss), correct / total))
        
        #validataion
        _, _, test_acc = Test(dicLoader['val_target'], student_model, criterion)
        total_acc = test_acc
        #continual val check
        for name in dicSourceName:
            if 'val_source' in name:
                _, _, source_acc = Test(dicLoader[name], student_model, criterion)
                total_acc += source_acc
            
        is_best_acc = total_acc > best_acc  
        if (epoch+1)%20 ==0 or is_best_acc:
            if is_best_acc : best_acc = total_acc
            is_best_acc = True
            best_acc = max(total_acc,best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': student_model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            },  args,
                filename = '{}_epoch_{}.pth.tar'.format(args.weigiht,epoch+1 if (epoch+1)%10==0 else ''),
                ACC_BEST=True
                )

            print('! ! ! save ! ! !')

