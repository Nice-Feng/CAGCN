import scipy.io as sio
import numpy as np
from sklearn.model_selection import KFold
from edge_select import edge_twosampels
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from GraphVAE import GraphEncoder
from DGNN import DGLayer
from GIN_layer import GIN_layer
from casual_cal import ginconv_cal
import torch.nn as nn
from torch.utils.data import random_split
import os
import random
import csv
import gc
import pickle
import random


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.01, lrpatience=5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.counter_lr = 0
        self.min_validation_loss = float('inf')
        self.min_validation_loss_lr = float('inf')
        self.lrpatience = lrpatience
        self.minlr = 0.0001

    def early_stop(self, validation_loss, lr):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if int(self.counter >= self.patience) | int(self.minlr >= lr):
                return True
        return False

    def decrease_lr(self, validtion_loss, lr):
        if validtion_loss < self.min_validation_loss_lr:
            self.min_validation_loss_lr = validtion_loss
            self.counter_lr = 0
        elif validtion_loss > (self.min_validation_loss_lr):
            self.counter_lr += 1
            if self.counter_lr >= self.lrpatience:
                lr = lr * 0.5
        return lr


class REC(nn.Module):
    def adjust_lr(self, epoch):
        self.lr = 0.01 * (0.1 ** (epoch // 8))

    def load_moudle(self, device):
        args = []
        input_dim = 19
        hidden_dim = 32
        output_dim = 2
        self.lr = 0.01
        self.encoder = GraphEncoder(input_dim, hidden_dim, device).to(device)
        self.gin_layer = GIN_layer(input_dim, hidden_dim, device).to(device)
        self.class_model = DGLayer(hidden_dim, output_dim, args, device).to(device)
        self.params = list(self.encoder.parameters()) + list(self.gin_layer.parameters()) + list(
            self.class_model.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.bestevalacc = 0
        self.besttestacc = 0

    def train_data(self, epochnum, device, train_loader, vali_loader, test_loader, early_stopper):
        self.encoder.train()
        self.gin_layer.train()
        self.class_model.train()
        train_acc_all = []
        test_acc_all = []
        eval_acc_all = []

        for epoch in range(epochnum):
            # self.adjust_lr(epoch)
            acc_accum = 0
            loss_all = 0
            num = 0
            for batch in train_loader:
                subtop, sublast, aedge = self.encoder(batch, device)
                loss, y_out, labels, nodeemd = ginconv_cal(subtop, sublast, batch, aedge, self.gin_layer,
                                                           self.class_model, device)
                # optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_all += loss.data.cpu().detach().numpy()
                pred = y_out.max(1, keepdim=True)[1]
                correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
                acc = correct / float(len(batch.y))

                acc_accum = acc_accum + acc
                num = num + 1
            acc_train = acc_accum / num

            print("accuracy train: %f" % (acc_train))
            print('Training epoch: {}'.format(epoch))

            self.lr = early_stopper.decrease_lr(loss_all, self.lr)
            print('Learning rate: {}'.format(self.lr))
            print('Loss:{}'.format(loss_all))
            train_acc_all = np.append(train_acc_all, acc_train)
            acc_eval, eval_edge_indexall, eval_aedge_all, eval_nodemb = self.evaluate_data(vali_loader, self.encoder,
                                                                                           self.gin_layer,
                                                                                           self.class_model)
            print("accuracy evaluation: %f" % (acc_eval))
            # judge best from epochs
            if acc_eval > self.bestevalacc:
                self.bestevalacc = acc_eval
                best_eval_edgeindex = eval_edge_indexall
                best_eval_weight = eval_aedge_all
                best_eval_node = eval_nodemb
            acc_test, test_edge_indexall, test_aedge_all, test_nodemb = self.test(test_loader, self.encoder,
                                                                                  self.gin_layer, self.class_model)
            print("accuracy test: %f" % (acc_test))
            # judge best from epochs
            if acc_test > self.besttestacc:
                self.besttestacc = acc_test
                best_test_edgeindex = test_edge_indexall
                best_test_weight = test_aedge_all
                best_test_node = test_nodemb
            test_acc_all = np.append(test_acc_all, acc_test)
            eval_acc_all = np.append(eval_acc_all, acc_eval)

        return train_acc_all, test_acc_all, eval_acc_all, \
            self.bestevalacc, self.besttestacc, \
            best_eval_weight, best_eval_edgeindex, best_test_weight, best_test_edgeindex, best_eval_node, best_test_node

    def evaluate_data(self, vali_loader, encoder, gin_layer, class_model):
        acc_accum = 0
        num = 0
        aedge_all = []
        edge_indexall = []
        node_all = []
        for batch in vali_loader:
            subtop, sublast, aedge = encoder(batch, self.device)
            loss, y_out, labels, node_emb = ginconv_cal(subtop, sublast, batch, aedge, gin_layer, class_model,
                                                        self.device)
            pred = y_out.max(1, keepdim=True)[1]
            correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
            acc = correct / float(len(batch.y))
            acc_accum = acc_accum + acc
            num = num + 1
            aedge_all.append(aedge.cpu().detach().numpy())
            edge_indexall.append(batch[0].edge_index)
            node_all.append(np.squeeze(node_emb.cpu().detach().numpy()))
        acc_eval = acc_accum / num
        node_all = np.array(node_all)
        return acc_eval, edge_indexall, aedge_all, node_all

    def test(self, test_loader, encoder, gin_layer, class_model):
        acc_accum = 0
        num = 0
        aedge_all = []
        edge_indexall = []
        node_all = []
        for batch in test_loader:
            subtop, sublast, aedge = encoder(batch, self.device)
            loss, y_out, labels, node_emb = ginconv_cal(subtop, sublast, batch, aedge, gin_layer, class_model,
                                                        self.device)
            pred = y_out.max(1, keepdim=True)[1]
            correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
            acc = correct / float(len(batch.y))
            acc_accum = acc_accum + acc
            num = num + 1
            aedge_all.append(aedge.cpu().detach().numpy())
            edge_indexall.append(batch[0].edge_index)
            node_all.append(np.squeeze(node_emb.cpu().detach().numpy()))
        acc_test = acc_accum / num
        node_all = np.array(node_all)
        return acc_test, edge_indexall, aedge_all, node_all


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for sub in range(7, 16):
        gdata_list = []
        sizeedg = []
        tname = r'.../rest_sub' + str(sub) + '.mat'
        task = sio.loadmat(tname)
        data = task['data']
        edge = task['edge']
        label = task['label'] - 1
        edge_top, edge_last = edge_twosampels(edge)  # 找到前50% and 后50%
        edge_top = np.mean(edge_top, axis=3)
        edge_last = np.mean(edge_last, axis=3)
        for sp in range(0, edge.shape[0]):
            matrix_tri = np.triu(edge_top[sp])
            positions = matrix_tri > 0.5
            indices = np.nonzero(positions)
            a = np.expand_dims(indices[0], axis=-1)
            b = np.expand_dims(indices[1], axis=-1)
            tmp_edgtop = np.concatenate((a, b), axis=1).T
            #
            matrix_tri = np.triu(edge_last[sp])
            positions = matrix_tri > 0.5
            indices = np.nonzero(positions)
            a_last = np.expand_dims(indices[0], axis=-1)
            b_last = np.expand_dims(indices[1], axis=-1)
            tmp_edglast = np.concatenate((a_last, b_last), axis=1).T
            gdata_top_last = Data(x=torch.tensor(data[sp]), y=torch.tensor(int(label[sp])),
                                  edge_index=torch.tensor(tmp_edgtop), lastedg_index=torch.tensor(tmp_edglast))
            gdata_list.append(gdata_top_last)
        ##
        arr = np.linspace(1, len(gdata_list), len(gdata_list))
        f_1 = open('acc_train' + str(sub) + '.csv', 'w', encoding='utf8', newline='')
        train_writter = csv.writer(f_1)
        f_2 = open('acc_validation' + str(sub) + '.csv', 'w', encoding='utf8', newline='')
        eval_writter = csv.writer(f_2)
        f_3 = open('acc_test' + str(sub) + '.csv', 'w', encoding='utf8', newline='')
        test_writter = csv.writer(f_3)
        gc.collect()
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        ind = 0
        for trainind, test_ind in kfold.split(gdata_list):
            test = [gdata_list[int(i)] for i in test_ind]
            eval = [gdata_list[int(i)] for i in test_ind]
            train = [gdata_list[int(i)] for i in trainind]
            mod = REC()
            mod.load_moudle(device)
            num_eval = len(eval)
            num_test = len(test)
            dataloader = dict()
            dataloader['train'] = DataLoader(train, batch_size=16, shuffle=True)
            dataloader['eval'] = DataLoader(eval, batch_size=1, shuffle=False)
            dataloader['test'] = DataLoader(test, batch_size=1, shuffle=False)
            early_stopper = EarlyStopper(patience=10, min_delta=0.001, lrpatience=5)
            train_acc_all, test_acc_all, eval_acc_all, bestevalacc, besttestacc, \
                best_eval_weight, best_eval_edgeindex, best_test_weight, best_test_edgeindex, best_eval_node, best_test_node \
                = mod.train_data(epochnum=30, device=device, train_loader=dataloader['train'], \
                                 vali_loader=dataloader['eval'], test_loader=dataloader['test'], \
                                 early_stopper=early_stopper)
            train_writter.writerow([train_acc_all])
            test_writter.writerow([test_acc_all])
            eval_writter.writerow([eval_acc_all])

            save_variable = {'besttestacc': besttestacc, 'bestevalacc': bestevalacc, \
                             'best_eval_weight': best_eval_weight, 'best_eval_edgeindex': best_eval_edgeindex, \
                             'best_test_weight': best_test_weight, 'best_test_edgeindex': best_test_edgeindex, \
                             'best_eval_node': best_eval_node, 'best_test_node': best_test_node}
            file = open('subgraph' + str(sub) + '_fold' + str(ind) + '.pickle', 'wb')
            ind = ind + 1
            pickle.dump(save_variable, file)
            file.close()
        f_1.close()
        f_2.close()
        f_3.close()




























