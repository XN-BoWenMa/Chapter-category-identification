import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from utils_me import get_max_lengths, get_evaluation
from dataset_me import MyDataset
from hierarchical_att_model_me import HierAttNet
import argparse
import shutil
import numpy as np
import random

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def get_args():
    path = "..\\..\\data\\output\\dl_file\\"
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoches", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--word_hidden_size", type=int, default=100)
    parser.add_argument("--sent_hidden_size", type=int, default=100)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=3,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default=path+"around_title_3_train.csv")
    parser.add_argument("--test_set", type=str, default=path+"around_title_3_test.csv")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default=path+"glove.6B.100d.txt")
    parser.add_argument("--n_filters", type=int, default=50)
    parser.add_argument("--filter_sizes", type=list, default=[1,2,3])
    parser.add_argument("--filter_sizes_fuse", type=list, default=[1,2,3,4,5,6])
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
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": False,
                       "num_workers":4,
                       "pin_memory":True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "num_workers":4,
                   "pin_memory":True,
                   "drop_last": False}
    max_word_length, max_sent_length = get_max_lengths(opt.train_set)
    # print(max_word_length)
    # print(max_sent_length)
    training_set = MyDataset(opt.train_set, opt.word2vec_path, max_sent_length, max_word_length)
    training_generator = DataLoaderX(training_set, **training_params)
    test_set = MyDataset(opt.test_set, opt.word2vec_path, max_sent_length, max_word_length)
    test_generator = DataLoaderX(test_set, **test_params)
    model = HierAttNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size, training_set.num_classes,
                       opt.word2vec_path, max_sent_length, max_word_length, opt.n_filters, opt.filter_sizes, opt.filter_sizes_fuse, opt.dropout)
    checkpoint = torch.load(opt.saved_path + os.sep + "around_title_3_new.pkl")
    model.load_state_dict(checkpoint['model'])
    if torch.cuda.is_available():
        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    model.eval()
    loss_ls = []
    te_label_ls = []
    te_pred_ls = []
    for te_label, te_title, te_before1, te_before2, te_before3, te_after1, te_after2, te_after3 in test_generator:
        num_sample = len(te_label)
        if torch.cuda.is_available():
            te_label = te_label.cuda()
            te_title = te_title.cuda()
            te_before1 = te_before1.cuda()
            te_before2 = te_before2.cuda()
            te_before3 = te_before3.cuda()
            te_after1 = te_after1.cuda()
            te_after2 = te_after2.cuda()
            te_after3 = te_after3.cuda()
        with torch.no_grad():
            model._init_hidden_state(num_sample)
            te_predictions = model(te_title, te_before1, te_before2, te_before3, te_after1, te_after2, te_after3, 6, 6)
        te_loss = criterion(te_predictions, te_label)
        loss_ls.append(te_loss * num_sample)
        te_label_ls.extend(te_label.clone().cpu())
        te_pred_ls.append(te_predictions.clone().cpu())
    te_loss = sum(loss_ls) / test_set.__len__()
    te_pred = torch.cat(te_pred_ls, 0)
    te_label = np.array(te_label_ls)
    test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["report"])
    print(te_loss.cpu().data)
    print(test_metrics["report"])
    
if __name__ == "__main__":
    opt = get_args()
    train(opt)