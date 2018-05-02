
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import lf
from lf import lf_dataset, lf_loss
from lf.lf_dataset import LfDataset
from lf.line_follower import LineFollower
from utils.dataset_wrapper import DatasetWrapper
from utils.dataset_parse import load_file_list

import numpy as np
import cv2
import sys
import json
import os
import yaml

with open(sys.argv[1]) as f:
    config = yaml.load(f)

sol_network_config = config['network']['sol']
pretrain_config = config['pretraining']

training_set_list = load_file_list(pretrain_config['training_set'])

train_dataset = LfDataset(training_set_list,
                          augmentation=True)
train_dataloader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True, num_workers=0,
                              collate_fn=lf_dataset.collate)
batches_per_epoch = int(pretrain_config['lf']['images_per_epoch']/pretrain_config['lf']['batch_size'])
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

test_set_list = load_file_list(pretrain_config['validation_set'])
test_dataset = LfDataset(test_set_list)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False, num_workers=0,
                             collate_fn=lf_dataset.collate)


line_follower = LineFollower()
line_follower.cuda()
optimizer = torch.optim.Adam(line_follower.parameters(), lr=pretrain_config['lf']['learning_rate'])

dtype = torch.cuda.FloatTensor

lowest_loss = np.inf
cnt_since_last_improvement = 0
for epoch in xrange(1000):
    print "Epoch", epoch
    sum_loss = 0.0
    steps = 0.0
    line_follower.train()
    for x in train_dataloader:
        #Only single batch for now
        x = x[0]

        positions = [Variable(x_i.type(dtype), requires_grad=False)[None,...] for x_i in x['lf_xyrs']]
        xy_positions = [Variable(x_i.type(dtype), requires_grad=False)[None,...] for x_i in x['lf_xyxy']]
        img = Variable(x['img'].type(dtype), requires_grad=False)[None,...]

        #There might be a way to handle this case later,
        #but for now we will skip it
        if len(xy_positions) <= 1:
            continue

        reset_interval = 4
        grid_line, _, _, xy_output = line_follower(img, positions[:1], steps=len(positions), all_positions=positions,
                                           reset_interval=reset_interval, randomize=True, skip_grid=True)

        loss = lf_loss.point_loss(xy_output, xy_positions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.data[0]
        steps += 1

    print "Train Loss", sum_loss/steps
    print "Real Epoch", train_dataloader.epoch

    sum_loss = 0.0
    steps = 0.0
    line_follower.eval()
    for x in test_dataloader:
        x = x[0]

        positions = [Variable(x_i.type(dtype), requires_grad=False, volatile=True)[None,...] for x_i in x['lf_xyrs']]
        xy_positions = [Variable(x_i.type(dtype), requires_grad=False, volatile=True)[None,...] for x_i in x['lf_xyxy']]
        img = Variable(x['img'].type(dtype), requires_grad=False, volatile=True)[None,...]

        if len(xy_positions) <= 1:
            continue

        grid_line, _, _, xy_output = line_follower(img, positions[:1], steps=len(positions), skip_grid=True)

        # line = torch.nn.functional.grid_sample(img.transpose(2,3), grid_line)
        # line = (line + 1.0) * 128
        # cv2.imwrite("tra/{}.png".format(steps), line.data[0].cpu().numpy().transpose())

        loss = lf_loss.point_loss(xy_output, xy_positions)

        sum_loss += loss.data[0]
        steps += 1

    cnt_since_last_improvement += 1
    if lowest_loss > sum_loss/steps:
        cnt_since_last_improvement = 0
        lowest_loss = sum_loss/steps
        print "Saving Best"

        if not os.path.exists(pretrain_config['snapshot_path']):
            os.makedirs(pretrain_config['snapshot_path'])

        torch.save(line_follower.state_dict(), os.path.join(pretrain_config['snapshot_path'], 'lf.pt'))

    print "Test Loss", sum_loss/steps, lowest_loss
    print ""

    if cnt_since_last_improvement >= pretrain_config['lf']['stop_after_no_improvement']:
        break
