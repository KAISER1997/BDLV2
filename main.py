from __future__ import division
import random
import numpy as np
import torch
import torch.nn as nn
from data_utils import cifar, ImageDataset
from contextloader import ContextSampler
import torch.utils.data as data
import torchvision.transforms as transforms
from expert import synth_expert
from net import *
import argparse
import json
import math
import os
import random
import shutil
import time
import torch.backends.cudnn as cudnn
from lib.utils import AverageMeter, accuracy

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
num_classes = len(class_names)
class2idx = {class_name: idx for idx, class_name in enumerate(class_names)}
idx2class = {idx: class_name for class_name, idx in class2idx.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate(
             test_loader,
             test_context,
             model_net,
             model_encoder,
             model_rejector,
             expert_fns,
             loss_fn,
             n_classes,
             context_partion_batchsize,
             device,
             config):
    '''
    Computes metrics for deferal
    -----
    Arguments:
    model_net: classifier model
    model_encoder: Context Encoder
    model_rejector: Rejector
    expert_fns: list of expert objects
    n_classes: number of classes
    test_loader: data loader for test or val
    test_context: X and Y context fixed
    context_partion_batchsize: BatchSize for breaking context before passing it through feature extractor to prevent GPU memory overflow
    '''

    classifier_correct=0
    classifier_total=0
    #  === Individual Expert X classifier Accuracies === #
    expert_correct_dic = {k: 0 for k in range(len(expert_fns))}
    expert_total_dic = {k: 0 for k in range(len(expert_fns))}
    #  === Individual  Expert X classifier Accuracies === #
    context_x,context_y=test_context
    context_x,context_y=context_x.to(device),context_y.to(device)

    cls_pred = []
    cls_true = []
    cls_correct_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}
    cls_true_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    cls_acc_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    expert_clswise_correct_list=np.zeros((len(expert_fns),10))
    expert_clswise_acc_list=np.zeros((len(expert_fns),10))
    # for j in range(len(expert_fns)):
    #     expert_clswise_correct_dic={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    #     expert_clswise_correct_list.append(expert_clswise_correct_dic)

    #     expert_clswise_acc_dic={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    #     expert_clswise_acc_list.append(expert_clswise_acc_dic)


    with torch.no_grad():
        context=Transform_context(context_x,model_net,context_partion_batchsize),context_y # context_x,context_y
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels[:,0].to(device)
            cls_true.extend(labels.data.cpu().numpy())
            logits = model_net(images) #B X 10

            _, predicted = torch.max(logits.data, 1)
            batch_size = logits.size()[0]  # batch_size
            cls_pred.extend(predicted.data.cpu().numpy())
            classifier_correct=classifier_correct+torch.sum(predicted==labels)
            classifier_total=classifier_total+batch_size


            for i, fn in enumerate(expert_fns, 0):
                expert=fn
                expert_context_x,expert_context_y,expert_context_m=Sample_Expert_context(expert,context) #Note that the batch size maybe bigger than what you can directly pass to NN
                full_context=torch.cat([expert_context_x,expert_context_y.unsqueeze(1),expert_context_m.unsqueeze(1).to(device)],1)
                encoded_context=model_encoder(full_context.float()).mean(0)
                rejector_logits=model_rejector(logits.float(),encoded_context.unsqueeze(0).expand(logits.shape[0],-1)) #Batch x 1
                combo_logits=torch.cat([logits,rejector_logits],1)
                # print("check-1 Combo shape B X 11",combo_logits.shape)
                final_selection=torch.argmax(combo_logits,1).long()
                mask= final_selection==n_classes
                final_selection[mask]=expert.predict(None,labels.long())[mask]
                expert_correct_dic[i]=expert_correct_dic[i]+torch.sum(final_selection==labels)
                expert_total_dic[i]=expert_total_dic[i]+labels.shape[0]

                for u in range(labels.shape[0]):
                    expert_clswise_correct_list[i,labels[u].item()]= expert_clswise_correct_list[i,labels[u].item()]+int(labels[u].item()==final_selection[u].item())


                

    #  === Individual Expert X Classifier Accuracies === #
    expert_accuracies = {"expert_{}".format(str(k)): 100 * expert_correct_dic[k] / (expert_total_dic[k] ) for k
                         in range(len(expert_fns))}
    classifier_accuracy=classifier_correct*100/classifier_total
    total_exp_acc=0
    for k in range(len(expert_fns)):
        total_exp_acc=total_exp_acc+expert_accuracies['expert_'+str(k)]
    mean_exp_acc=total_exp_acc/len(expert_fns)

    for j in range(len(cls_true)):
        # print("ora",len(cls_true))
        cls_correct_dict[cls_true[j]]=cls_correct_dict[cls_true[j]]+int(cls_true[j]==cls_pred[j])
        cls_true_dict[cls_true[j]]=cls_true_dict[cls_true[j]]+1
    
    for i in range(10):
        cls_acc_dict[i]=100*cls_correct_dict[i]/cls_true_dict[i]
        for j in range(len(expert_fns)):
            expert_clswise_acc_list[j,i]=expert_clswise_correct_list[j,i]/cls_true_dict[i]




    # Add expert accuracies dict
    to_print = {"classifier_accuracy":classifier_accuracy,"Mean_exp_acc":mean_exp_acc,**expert_accuracies}
    print("Classwise acc solo classifier ",cls_acc_dict)
    print("Classwise With Clsfr X expert-")
    for j in range(len(expert_fns)):
        print("Expert-"+str(j),expert_clswise_acc_list[j])
    print(to_print, flush=True)
    return to_print

def Transform_context(context_x,model_net,context_partion_batchsize):
    num_partitions=context_x.shape[0]//context_partion_batchsize
    context_list=[]
    for i in range(num_partitions):
        with torch.no_grad():
            # print(context_x.shape,"cotntext")
            context_feat=model_net(context_x[i*context_partion_batchsize:(i+1)*context_partion_batchsize,:,:,:])
            context_list.append(context_feat)
    
    return torch.cat(context_list,0)


def Sample_Expert_context(expert,context):
    context_x,context_y=context
    context_m=expert.predict(context_x,context_y)
    return(context_x,context_y,context_m)










def train_epoch(iters,
                warmup_iters,
                lrate,
                train_loader,
                # context_loader,
                model_net,
                model_encoder,
                model_rejector,
                optimizer,
                scheduler,
                epoch,
                expert_fns,
                loss_fn,
                n_classes,
                # n_experts,
                alpha,
                fixed_context,
                config):
    """ Train for one epoch """
    device=config["device"]
    context_partion_batchsize=10
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model_net.train()
    model_encoder.train()
    model_rejector.train()

    end = time.time()

    epoch_train_loss = []
    # contextloader_iterator = iter(context_loader)

    for i, (input, target) in enumerate(train_loader):
        context_x,context_y=fixed_context

        # try:
        #     context_x,context_y = next(contextloader_iterator)
        # except StopIteration:
        #     contextloader_iterator = iter(context_loader)
        #     context_x,context_y = next(contextloader_iterator)



        if iters < warmup_iters:
            lr = lrate * float(iters) / warmup_iters
            # print(iters, lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        target = target[:,0].to(device)
        input = input.to(device)
        # print(input.shape,target,"hellp")
        rej_target=torch.zeros(target.shape).to(device)+n_classes

        # compute output
        logits = model_net(input) #B X 10
        context=Transform_context(context_x,model_net,context_partion_batchsize),context_y # context_x,context_y
        total_loss=0
        for j in range(len(expert_fns)):
            expert=expert_fns[j]
            expert_context_x,expert_context_y,expert_context_m=Sample_Expert_context(expert,context) #Note that the batch size is bigger than what you can directly pass to NN
            full_context=torch.cat([expert_context_x,expert_context_y.unsqueeze(1),expert_context_m.unsqueeze(1).to(device)],1)
            encoded_context=model_encoder(full_context.float()).mean(0)
            rejector_logits=model_rejector(logits.float(),encoded_context.unsqueeze(0).expand(logits.shape[0],-1)) #Batch x 1
            combo_logits=torch.cat([logits,rejector_logits],1)
            # print(loss_fn(combo_logits,rej_target.long()).shape,"DDSDS")
            lo=loss_fn(combo_logits,target)+(expert.predict(None,target)==target)*loss_fn(combo_logits,rej_target.long())
            # print(lo.shape,'fadd')
            loss=lo.mean()
            total_loss=total_loss+loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        batch_size = input.size()[0]  # batch_size


        epoch_train_loss.append(total_loss.item())

        if not iters < warmup_iters:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        iters += 1

        if i % 100 == 0:
            print("Total Loss every 100 Runs-",total_loss.item())

    return iters, np.average(epoch_train_loss)


def train(model_net,
          model_encoder,
          model_rejector,
          train_dataset,
          fixed_context,
          validation_dataset,
          expert_fns,
          device,
          config):
    n_classes = config["n_classes"] 
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["batch_size"], shuffle=True, drop_last=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(validation_dataset,
                                               batch_size=config["batch_size"], shuffle=True, drop_last=True, **kwargs)
    model_net = model_net.to(device)
    model_encoder = model_encoder.to(device)
    model_rejector = model_rejector.to(device)

    cudnn.benchmark = True
    params = list(model_net.parameters()) + list(model_encoder.parameters())+list(model_rejector.parameters())
    optimizer = torch.optim.SGD(params, config["lr"],
                                momentum=0.9, nesterov=True,
                                weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss_fn = criterion #getattr(criterion, config["loss_type"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader) * config["epochs"])
    best_validation_acc = -np.inf
    patience = 0
    iters = 0
    warmup_iters = config["warmup_epochs"] * len(train_loader)
    lrate = config["lr"]
    context_partion_batchsize=10

    for epoch in range(0, config["epochs"]):
        iters, train_loss = train_epoch(iters,
                                        warmup_iters,
                                        lrate,
                                        train_loader,
                                        # context_loader,
                                        model_net,
                                        model_encoder,
                                        model_rejector,
                                        optimizer,
                                        scheduler,
                                        epoch,
                                        expert_fns,
                                        loss_fn,
                                        n_classes,
                                        # n_experts,
                                        config["alpha"],
                                        fixed_context,
                                        config)
        metrics =  evaluate(
                            valid_loader,
                            fixed_context,
                            model_net,
                            model_encoder,
                            model_rejector,
                            expert_fns,
                            loss_fn,
                            n_classes,
                            context_partion_batchsize,
                            device,
                            config)

        validation_acc = metrics["Mean_exp_acc"]

        if validation_acc > best_validation_acc:
            best_validation_acc = validation_acc
            print("Saving the model with mean validation accuracy {}".format(
                metrics['Mean_exp_acc']), flush=True)
            save_path = os.path.join(config["ckp_dir"],
                                     config["experiment_name"] + '_' + str(len(expert_fns)) + '_experts')
            # torch.save(model.state_dict(), save_path + '.pt')
            # Additionally save the whole config dict
            with open(save_path + '.json', "w") as f:
                json.dump(config, f)
            patience = 0
        else:
            patience += 1

        if patience >= config["patience"]:
            print("Early Exiting Training.", flush=True)
            break







def basic_expt(config):
    config["ckp_dir"] = "./" + config["loss_type"] + "_basic"
    os.makedirs(config["ckp_dir"], exist_ok=True)

    trainD, valD = cifar.read(test=False, only_id=True, data_aug=True)
    normalize = transforms.Normalize(mean = trainD.means, std = trainD.stds)
    # print(trainD.means)

    contextD=ImageDataset(imgs=trainD.data,targets=trainD.labels[:,0],img_transform=transforms.Compose([
                        normalize,]))

    n= 3
    print("Training for n={}".format(n))
    num_experts = n
    expert_fns=[synth_expert(0,10,10) ,synth_expert(0,5,10),synth_expert(0,1,10)]
    trainD, valD = cifar.read(test=False, only_id=True, data_aug=True)


    N_WAY = 10 #Number of classes
    K_SHOT = 50
    device='cuda:0'

    context_data_loader = data.DataLoader(contextD,
                                        batch_sampler=ContextSampler(contextD.targets,
                                                                        include_query=False,
                                                                        N_way=N_WAY,
                                                                        K_shot=K_SHOT,
                                                                        shuffle=False),
                                            )

    fixed_context=next(iter(context_data_loader))
    fixed_context[0]=fixed_context[0].to(device)
    fixed_context[1]=fixed_context[1].to(device)
    model_net=ResNet18()
    model_encoder=ENCODER()
    model_rejector=REJECTOR()
    train(model_net,model_encoder,model_rejector,trainD,fixed_context,valD,expert_fns,device,config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="scaling parameter for the loss function, default=1.0.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20,
                        help="number of patience steps for early stopping the training.")
    parser.add_argument("--expert_type", type=str, default="predict",
                        help="specify the expert type. For the type of experts available, see-> models -> experts. defualt=predict.")
    parser.add_argument("--n_classes", type=int, default=10,
                        help="K for K class classification.")
    parser.add_argument("--k", type=int, default=5)
    # Dani experiments =====
    parser.add_argument("--n_experts", type=int, default=2)
    # Dani experiments =====
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--loss_type", type=str, default="softmax",
                        help="surrogate loss type for learning to defer.")
    parser.add_argument("--ckp_dir", type=str, default="./Models",
                        help="directory name to save the checkpoints.")
    parser.add_argument("--experiment_name", type=str, default="exp1",
                        help="specify the experiment name. Checkpoints will be saved with this name.")

    config = parser.parse_args().__dict__

    print(config)
    # increase_experts(config)
    basic_expt(config)












