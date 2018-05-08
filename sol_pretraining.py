
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import sol
from sol.start_of_line_finder import StartOfLineFinder
from sol.alignment_loss import alignment_loss
from sol.sol_dataset import SolDataset
from sol.crop_transform import CropTransform

from utils.dataset_wrapper import DatasetWrapper
from utils.dataset_parse import load_file_list

import numpy as np
import cv2
import json
import yaml
import sys
import os
import math

from utils import transformation_utils, drawing

with open(sys.argv[1]) as f:
    config = yaml.load(f)

sol_network_config = config['network']['sol']
pretrain_config = config['pretraining']

training_set_list = load_file_list(pretrain_config['training_set'])
train_dataset = SolDataset(training_set_list,
                           rescale_range=pretrain_config['sol']['training_rescale_range'],
                           transform=CropTransform(pretrain_config['sol']['crop_params']))

train_dataloader = DataLoader(train_dataset,
                              batch_size=pretrain_config['sol']['batch_size'],
                              shuffle=True, num_workers=0,
                              collate_fn=sol.sol_dataset.collate)

batches_per_epoch = int(pretrain_config['sol']['images_per_epoch']/pretrain_config['sol']['batch_size'])
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

test_set_list = load_file_list(pretrain_config['validation_set'])
test_dataset = SolDataset(test_set_list,
                          rescale_range=pretrain_config['sol']['validation_rescale_range'],
                          transform=None)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=sol.sol_dataset.collate)


base0 = sol_network_config['base0']
base1 = sol_network_config['base1']
sol = StartOfLineFinder(base0, base1)
if torch.cuda.is_available():
    sol.cuda()
    dtype = torch.cuda.FloatTensor
else:
    print "Warning: Not using a GPU, untested"
    dtype = torch.FloatTensor

alpha_alignment = pretrain_config['sol']['alpha_alignment']
alpha_backprop = pretrain_config['sol']['alpha_backprop']

optimizer = torch.optim.Adam(sol.parameters(), lr=pretrain_config['sol']['learning_rate'])

lowest_loss = np.inf
cnt_since_last_improvement = 0
for epoch in xrange(1000):
    print "Epoch", epoch

    sol.train()
    sum_loss = 0.0
    steps = 0.0

    for step_i, x in enumerate(train_dataloader):
        img = Variable(x['img'].type(dtype), requires_grad=False)

        sol_gt = None
        if x['sol_gt'] is not None:
            # This is needed because if sol_gt is None it means that there
            # no GT positions in the image. The alignment loss will handle,
            # it correctly as None
            sol_gt = Variable(x['sol_gt'].type(dtype), requires_grad=False)


        predictions = sol(img)
        predictions = transformation_utils.pt_xyrs_2_xyxy(predictions)
        loss = alignment_loss(predictions, sol_gt, x['label_sizes'], alpha_alignment, alpha_backprop)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.data[0]
        steps += 1

    print "Train Loss", sum_loss/steps
    #print "Real Epoch", train_dataloader.epoch

    sol.eval()
    sum_loss = 0.0
    steps = 0.0

    for step_i, x in enumerate(test_dataloader):
        img = Variable(x['img'].type(dtype), requires_grad=False, volatile=True)
        sol_gt = Variable(x['sol_gt'].type(dtype), requires_grad=False, volatile=True)

        predictions = sol(img)
        predictions = transformation_utils.pt_xyrs_2_xyxy(predictions)
        loss = alignment_loss(predictions, sol_gt, x['label_sizes'], alpha_alignment, alpha_backprop)

        ### Write images to file to visualization
        #org_img = img[0].data.cpu().numpy().transpose([2,1,0])
        #org_img = ((org_img + 1)*128).astype(np.uint8)
        #org_img = org_img.copy()
        #org_img = drawing.draw_sol_torch(predictions, org_img)
        # cv2.imwrite("data/sol_val_2/{}.png".format(step_i), org_img)

        sum_loss += loss.data[0]
        steps += 1

    cnt_since_last_improvement += 1
    if lowest_loss > sum_loss/steps:
        cnt_since_last_improvement = 0
        lowest_loss = sum_loss/steps
        print "Saving Best"

        if not os.path.exists(pretrain_config['snapshot_path']):
            os.makedirs(pretrain_config['snapshot_path'])

        torch.save(sol.state_dict(), os.path.join(pretrain_config['snapshot_path'], 'sol.pt'))

    print "Test Loss", sum_loss/steps, lowest_loss
    print ""

    if cnt_since_last_improvement >= pretrain_config['sol']['stop_after_no_improvement']:
        break
