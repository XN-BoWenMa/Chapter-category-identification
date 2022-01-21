import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from utils_me import get_max_lengths, get_evaluation
from dataset_me import MyDataset
from hierarchical_att_model_me import HierAttNet
from torch.utils.data.sampler import WeightedRandomSampler
# from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np
import random

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)
 
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
 
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        # print("---------------regularization weight---------------")
        # for name ,w in weight_list:
        #     print(name)
        # print("---------------------------------------------------")

def get_args():
    path = "..\\..\\data\\output\\dl_file\\"
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoches", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--word_hidden_size", type=int, default=100)
    parser.add_argument("--sent_hidden_size", type=int, default=100)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default=path+"baseline_train.csv")
    parser.add_argument("--test_set", type=str, default=path+"baseline_valid.csv")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default=path+"glove.6B.100d.txt")
    # parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="..\..\data\output\dl_model_save")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.deterministic = True
    # output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    # output_file.write("Model's parameters: {}".format(vars(opt))) #返回属性和属性值
    max_word_length, max_sent_length = get_max_lengths(opt.train_set)
    # print(max_word_length)
    # print(max_sent_length)
    training_set = MyDataset(opt.train_set, opt.word2vec_path, max_sent_length, max_word_length)
    # weight_list = []
    # for i in range(0,5):
    #     weight = 1. / training_set.labels.count(i)
    #     weight_list.append(weight)
    # samples_weight = np.array([weight_list[t] for t in training_set.labels])
    # samples_weight = torch.from_numpy(samples_weight)
    # samples_weight = samples_weight.double()
    # sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "num_workers":1,
                       "pin_memory":True,
                       "drop_last": True}
    # training_params = {"batch_size": opt.batch_size,
    #                    "shuffle": False,
    #                    "num_workers":4,
    #                    "pin_memory":True,
    #                    "drop_last": True,
    #                    "sampler":sampler}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "num_workers":1,
                   "pin_memory":True,
                   "drop_last": False}
    training_generator = DataLoader(training_set, **training_params)
    test_set = MyDataset(opt.test_set, opt.word2vec_path, max_sent_length, max_word_length)
    test_generator = DataLoader(test_set, **test_params)
    model = HierAttNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size, training_set.num_classes,
                       opt.word2vec_path, max_sent_length, max_word_length, opt.dropout)
    # writer.add_graph(model, torch.zeros(opt.batch_size, max_sent_length, max_word_length))
    reg_loss = Regularization(model, opt.weight_decay, p=2)
    if torch.cuda.is_available():
        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, betas=(0.9, 0.99), weight_decay=0.0005)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, betas=(0.9, 0.99))
    # optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr) #效果不佳
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
    best_loss = 1e5
    best_epoch = 0
    model.train() #开始训练模式
    num_iter_per_epoch = len(training_generator) #即一共要训练多少个iter
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    learning_rate_descend = 0
    history_loss = []
    for epoch in range(opt.num_epoches):
        # training_generator = DataLoader(training_set, **training_params)
        prefetcher = data_prefetcher(training_generator)
        feature, label = prefetcher.next()
        iter = 0
        while feature is not None:
            iter += 1
        # for iter, (feature, label) in enumerate(training_generator):
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            model._init_hidden_state() #初始化隐藏层
            predictions = model(feature)
            loss = criterion(predictions, label)
            if opt.weight_decay>0:
                loss = loss + reg_loss(model)
            # history_loss.append(loss) #保存历史loss值
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if (iter) % 10 == 0:
                training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
                print("Train:Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                    epoch + 1,
                    opt.num_epoches,
                    iter,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    loss, training_metrics["accuracy"]))
            else:
                print("Train:Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}".format(
                    epoch + 1,
                    opt.num_epoches,
                    iter,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    loss))
            feature, label = prefetcher.next()
        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature, te_label in test_generator:
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_feature = te_feature.cuda()
                    te_label = te_label.cuda()
                with torch.no_grad():
                    model._init_hidden_state(num_sample)
                    te_predictions = model(te_feature)
                te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.append(te_predictions.clone().cpu())
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy"])
            print("Test:Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"]))
            model.train() #注意训练完之后，回到model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch              
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, opt.saved_path + os.sep + "basic_model.pkl")
            else:
                # learning_rate_descend += 1
                # if learning_rate_descend % 2 == 0:
                #     scheduler.step()
                #     learning_rate_descend = 0
                scheduler.step()
            # if (epoch+1)==2:
            #     optimizer.param_groups[0]['lr'] = 0.001
            # else:
            #     if optimizer.param_groups[0]['lr']==0.001:
            #         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
            #     if optimizer.param_groups[0]['lr']==0.0005:
            #         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.2)
                # learning_rate_descend += 1
                # if learning_rate_descend % 2 == 0:           
            # Early stopping
            # 3轮后再没有将loss降下来，则停止训练
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break

if __name__ == "__main__":
    opt = get_args()
    train(opt)