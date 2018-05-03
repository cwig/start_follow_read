import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

from hw import hw_dataset
from hw import cnn_lstm
from hw.hw_dataset import HwDataset

from utils.dataset_wrapper import DatasetWrapper
from utils import safe_load

import numpy as np
import cv2
import sys
import json
import os
from utils import string_utils, error_rates
import time
import random
import yaml

from utils.dataset_parse import load_file_list

with open(sys.argv[1]) as f:
    config = yaml.load(f)

hw_network_config = config['network']['hw']
pretrain_config = config['pretraining']

char_set_path = hw_network_config['char_set_path']

with open(char_set_path) as f:
    char_set = json.load(f)

idx_to_char = {}
for k,v in char_set['idx_to_char'].iteritems():
    idx_to_char[int(k)] = v

training_set_list = load_file_list(pretrain_config['training_set'])
train_dataset = HwDataset(training_set_list,
                          char_set['char_to_idx'], augmentation=True,
                          img_height=hw_network_config['input_height'])

train_dataloader = DataLoader(train_dataset,
                             batch_size=pretrain_config['hw']['batch_size'],
                             shuffle=True, num_workers=0, drop_last=True,
                             collate_fn=hw_dataset.collate)

batches_per_epoch = int(pretrain_config['hw']['images_per_epoch']/pretrain_config['hw']['batch_size'])
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

test_set_list = load_file_list(pretrain_config['validation_set'])
test_dataset = HwDataset(test_set_list,
                         char_set['char_to_idx'],
                         img_height=hw_network_config['input_height'])

test_dataloader = DataLoader(test_dataset,
                             batch_size=pretrain_config['hw']['batch_size'],
                             shuffle=False, num_workers=0,
                             collate_fn=hw_dataset.collate)



criterion = CTCLoss()

hw = cnn_lstm.create_model(hw_network_config)
hw.cuda()


optimizer = torch.optim.Adam(hw.parameters(), lr=pretrain_config['hw']['learning_rate'])
dtype = torch.cuda.FloatTensor

lowest_loss = np.inf
cnt_since_last_improvement = 0
for epoch in xrange(1000):
    print "Epoch", epoch
    sum_loss = 0.0
    steps = 0.0
    hw.train()
    for i, x in enumerate(train_dataloader):

        line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
        labels =  Variable(x['labels'], requires_grad=False)
        label_lengths = Variable(x['label_lengths'], requires_grad=False)

        preds = hw(line_imgs).cpu()

        output_batch = preds.permute(1,0,2)
        out = output_batch.data.cpu().numpy()

        for i, gt_line in enumerate(x['gt']):
            logits = out[i,...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred, idx_to_char, False)
            cer = error_rates.cer(gt_line, pred_str)
            sum_loss += cer
            steps += 1


        batch_size = preds.size(1)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

        # print "before"
        loss = criterion(preds, labels, preds_size, label_lengths)
        # print "after"

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print "Train Loss", sum_loss/steps
    print "Real Epoch", train_dataloader.epoch

    sum_loss = 0.0
    steps = 0.0
    hw.eval()

    for x in test_dataloader:
        line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False, volatile=True)
        labels =  Variable(x['labels'], requires_grad=False, volatile=True)
        label_lengths = Variable(x['label_lengths'], requires_grad=False, volatile=True)

        preds = hw(line_imgs).cpu()

        output_batch = preds.permute(1,0,2)
        out = output_batch.data.cpu().numpy()

        for i, gt_line in enumerate(x['gt']):
            logits = out[i,...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred, idx_to_char, False)
            cer = error_rates.cer(gt_line, pred_str)
            sum_loss += cer
            steps += 1

    cnt_since_last_improvement += 1
    if lowest_loss > sum_loss/steps:
        cnt_since_last_improvement = 0
        lowest_loss = sum_loss/steps
        print "Saving Best"

        if not os.path.exists(pretrain_config['snapshot_path']):
            os.makedirs(pretrain_config['snapshot_path'])

        torch.save(hw.state_dict(), os.path.join(pretrain_config['snapshot_path'], 'hw.pt'))

    print "Test Loss", sum_loss/steps, lowest_loss
    print ""

    if cnt_since_last_improvement >= pretrain_config['hw']['stop_after_no_improvement'] and lowest_loss<0.9:
        break
