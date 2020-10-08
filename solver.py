"""
This script is to train the network, based on the code written by Kaituo Xu
Date: 2019.06
Author: Andong Li
"""
import torch.nn as nn
import torch
import argparse
import time
import os
from Backup import *
import numpy as np
import gc
import hdf5storage


tr_batch, tr_epoch, cv_epoch = [], [], []

class Solver(object):
    def __init__(self, data, model, optimizer, args):
        # load args parameters
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.loss_path = args.loss_path
        self.model = model
        self.optimizer = optimizer
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.save_path = args.save_path
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.best_path = args.best_path
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.print_freq = args.print_freq

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            # package is the loading model
            package = torch.load(self.continue_from)
            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[self.start_epoch] = package['cv_loss'][:self.start_epoch]

        else:
            self.start_epoch = 0

        os.makedirs(self.save_path, exist_ok=True)
        self.prev_cv_loss = float("inf")
        self.best_cv_loss = float("inf")
        self.cv_no_impv = 0
        self.having = False

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            print("Begin to train.....")
            self.model.train()
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 90)
            print("End of Epoch %d, Time: %4f s, Train_Loss:%5f" % (int(epoch+1), time.time()-start, tr_avg_loss))
            print('-' * 90)

            # save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_path, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.serialize(self.model,
                                                self.optimizer, epoch + 1,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Cross cv
            print("Begin Cross Validation....")
            self.model.eval()    # BN and Dropout is off
            cv_avg_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 90)
            print("Time: %4fs, CV_Loss:%5f" % (time.time() - start, cv_avg_loss))
            print('-' * 90)

            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = cv_avg_loss
            tr_epoch.append(tr_avg_loss)
            cv_epoch.append(cv_avg_loss)

            # save loss
            loss = {}
            loss['tr_loss'] = tr_epoch
            loss['cv_loss'] = cv_epoch
            hdf5storage.savemat(self.loss_path, loss)

            # Adjust learning rate and early stop
            if self.half_lr:
                if cv_avg_loss >= self.prev_cv_loss:
                    self.cv_no_impv += 1
                    if self.cv_no_impv == 3:
                        self.having = True
                    if self.cv_no_impv >= 5 and self.early_stop == True:
                        print("No improvement and apply early stop")
                        break
                else:
                    self.cv_no_impv = 0

            if self.having == True:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to %5f' % (optim_state['param_groups'][0]['lr']))
                self.having = False
            self.prev_cv_loss = cv_avg_loss

            if cv_avg_loss < self.best_cv_loss:
                self.best_cv_loss = cv_avg_loss
                torch.save(self.model.state_dict(), self.best_path)
                print("Find better cv model, saving to %s" % os.path.split(self.best_path)[1])

    def _run_one_epoch(self, epoch, cross_valid = False):
        start1 = time.time()
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        for batch_id, batch_info in enumerate(data_loader.get_data_loader()):
            batch_feat = batch_info.feats.cuda()
            batch_label = batch_info.labels.cuda()
            batch_frame_mask_list = batch_info.frame_mask_list
            esti_out_list = self.model(batch_feat)
            batch_loss = mse_loss_stage(esti_out_list, batch_label, batch_frame_mask_list)
            tr_batch.append(batch_loss.item())

            if not cross_valid:
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            total_loss += batch_loss.item()
            gc.collect()
            if batch_id % self.print_freq == 0:
                print("Epoch:%d, Iter:%d, Average_loss:%5f,Current_loss:%5f, %d ms/batch."
                      % (int(epoch+1), int(batch_id), total_loss / (batch_id+1), batch_loss.item(),
                         1000 * (time.time() - start1) / (batch_id + 1)))
        return total_loss / (batch_id + 1)