
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import sol
from sol import sol_dataset
from sol.start_of_line_finder import StartOfLineFinder
from sol.alignment_loss import alignment_loss
from sol.sol_dataset import SolDataset
from sol.crop_transform import CropTransform

from lf.line_follower import LineFollower
from hw import cnn_lstm

from utils.dataset_wrapper import DatasetWrapper
from utils import safe_load, transformation_utils

import numpy as np
import cv2
import json
import sys
import os
import time
import random
import yaml

from utils.continuous_state import init_model
from utils.dataset_parse import load_file_list

def training_step(config):

    train_config = config['training']

    allowed_training_time = train_config['sol']['reset_interval']
    init_training_time = time.time()

    training_set_list = load_file_list(train_config['training_set'])
    train_dataset = SolDataset(training_set_list,
                               rescale_range=train_config['sol']['training_rescale_range'],
                               transform=CropTransform(train_config['sol']['crop_params']))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_config['sol']['batch_size'],
                                  shuffle=True, num_workers=0,
                                  collate_fn=sol_dataset.collate)

    batches_per_epoch = int(train_config['sol']['images_per_epoch']/train_config['sol']['batch_size'])
    train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

    test_set_list = load_file_list(train_config['validation_set'])
    test_dataset = SolDataset(test_set_list,
                              rescale_range=train_config['sol']['validation_rescale_range'],
                              random_subset_size=train_config['sol']['validation_subset_size'],
                              transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=sol_dataset.collate)


    alpha_alignment = train_config['sol']['alpha_alignment']
    alpha_backprop = train_config['sol']['alpha_backprop']

    sol, lf, hw = init_model(config, only_load='sol')

    dtype = torch.cuda.FloatTensor

    lowest_loss = np.inf
    lowest_loss_i = 0
    epoch = -1
    while True:#This ends on a break based on the current itme
        epoch += 1
        print "Train Time:",(time.time() - init_training_time), "Allowed Time:", allowed_training_time

        sol.eval()
        sum_loss = 0.0
        steps = 0.0
        start_time = time.time()
        for step_i, x in enumerate(test_dataloader):
            img = Variable(x['img'].type(dtype), requires_grad=False)

            sol_gt = None
            if x['sol_gt'] is not None:
                sol_gt = Variable(x['sol_gt'].type(dtype), requires_grad=False)

            predictions = sol(img)
            predictions = transformation_utils.pt_xyrs_2_xyxy(predictions)
            loss = alignment_loss(predictions, sol_gt, x['label_sizes'], alpha_alignment, alpha_backprop)
            sum_loss += loss.data[0]
            steps += 1

        if epoch == 0:
            print "First Validation Step Complete"
            print "Benchmark Validation CER:", sum_loss/steps
            lowest_loss = sum_loss/steps

            sol, lf, hw = init_model(config, sol_dir='current', only_load='sol')

            optimizer = torch.optim.Adam(sol.parameters(), lr=train_config['sol']['learning_rate'])
            optim_path = os.path.join(train_config['snapshot']['current'], "sol_optim.pt")
            if os.path.exists(optim_path):
                print "Loading Optim Settings"
                optimizer.load_state_dict(safe_load.torch_state(optim_path))
            else:
                print "Failed to load Optim Settings"

        elif lowest_loss > sum_loss/steps:
            lowest_loss = sum_loss/steps
            print "Saving Best"

            dirname = train_config['snapshot']['best_validation']
            if not len(dirname) != 0 and os.path.exists(dirname):
                os.makedirs(dirname)

            save_path = os.path.join(dirname, "sol.pt")

            torch.save(sol.state_dict(), save_path)
            lowest_loss_i = epoch

        print "Test Loss", sum_loss/steps, lowest_loss
        print "Time:", time.time() - start_time
        print ""

        print "Epoch", epoch

        if allowed_training_time < (time.time() - init_training_time):
            print "Out of time. Saving current state and exiting..."
            dirname = train_config['snapshot']['current']
            if not len(dirname) != 0 and os.path.exists(dirname):
                os.makedirs(dirname)

            save_path = os.path.join(dirname, "sol.pt")
            torch.save(sol.state_dict(), save_path)

            optim_path = os.path.join(dirname, "sol_optim.pt")
            torch.save(optimizer.state_dict(), optim_path)
            break

        sol.train()
        sum_loss = 0.0
        steps = 0.0
        start_time = time.time()
        for step_i, x in enumerate(train_dataloader):
            img = Variable(x['img'].type(dtype), requires_grad=False)

            sol_gt = None
            if x['sol_gt'] is not None:
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
        print "Real Epoch", train_dataloader.epoch
        print "Time:", time.time() - start_time



if __name__ == "__main__":
    config_path = sys.argv[1]

    with open(config_path) as f:
        config = yaml.load(f)

    cnt = 0
    while True:
        print ""
        print "Full Step", cnt
        print ""
        cnt += 1
        training_step(config)
