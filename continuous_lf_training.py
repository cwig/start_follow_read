
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import sol
from sol.start_of_line_finder import StartOfLineFinder

import lf
from lf import lf_dataset
from lf.lf_dataset import LfDataset
from lf.line_follower import LineFollower
from lf import lf_loss

from lf.line_follower import LineFollower
from hw import cnn_lstm

from utils.dataset_wrapper import DatasetWrapper
from utils import safe_load

import numpy as np
import cv2
import sys
import json
import os
import time
import random
import yaml

from utils import string_utils, error_rates
from utils.continuous_state import init_model
from utils.dataset_parse import load_file_list

def training_step(config):

    char_set_path = config['network']['hw']['char_set_path']

    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k,v in char_set['idx_to_char'].iteritems():
        idx_to_char[int(k)] = v

    train_config = config['training']

    allowed_training_time = train_config['lf']['reset_interval']
    init_training_time = time.time()

    training_set_list = load_file_list(train_config['training_set'])
    train_dataset = LfDataset(training_set_list,
                              augmentation=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=1,
                                  shuffle=True, num_workers=0,
                                  collate_fn=lf_dataset.collate)
    batches_per_epoch = int(train_config['lf']['images_per_epoch']/train_config['lf']['batch_size'])
    train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

    test_set_list = load_file_list(train_config['validation_set'])
    test_dataset = LfDataset(test_set_list,
                             random_subset_size=train_config['lf']['validation_subset_size'])
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False, num_workers=0,
                                 collate_fn=lf_dataset.collate)

    _, lf, hw = init_model(config, only_load=['lf', 'hw'])
    hw.eval()

    dtype = torch.cuda.FloatTensor

    lowest_loss = np.inf
    lowest_loss_i = 0
    for epoch in xrange(10000000):
        lf.eval()
        sum_loss = 0.0
        steps = 0.0
        start_time = time.time()
        for step_i, x in enumerate(test_dataloader):
            if x is None:
                continue
            #Only single batch for now
            x = x[0]
            if x is None:
               continue

            positions = [Variable(x_i.type(dtype), requires_grad=False)[None,...] for x_i in x['lf_xyrs']]
            xy_positions = [Variable(x_i.type(dtype), requires_grad=False)[None,...] for x_i in x['lf_xyxy']]
            img = Variable(x['img'].type(dtype), requires_grad=False)[None,...]

            #There might be a way to handle this case later,
            #but for now we will skip it
            if len(xy_positions) <= 1:
                print "Skipping"
                continue

            grid_line, _, _, xy_output = lf(img, positions[:1], steps=len(positions), skip_grid=False)

            line = torch.nn.functional.grid_sample(img.transpose(2,3), grid_line)
            line = line.transpose(2,3)
            predictions = hw(line)

            out = predictions.permute(1,0,2).data.cpu().numpy()
            gt_line = x['gt']
            pred, raw_pred = string_utils.naive_decode(out[0])
            pred_str = string_utils.label2str_single(pred, idx_to_char, False)
            cer = error_rates.cer(gt_line, pred_str)
            sum_loss += cer
            steps += 1

            # l = line[0].transpose(0,1).transpose(1,2)
            # l = (l + 1)*128
            # l_np = l.data.cpu().numpy()
            #
            # cv2.imwrite("example_line_out.png", l_np)
            # print "Saved!"
            # raw_input()

            # loss = lf_loss.point_loss(xy_output, xy_positions)
            #
            # sum_loss += loss.data[0]
            # steps += 1

        if epoch == 0:
            print "First Validation Step Complete"
            print "Benchmark Validation Loss:", sum_loss/steps
            lowest_loss = sum_loss/steps

            _, lf, _ = init_model(config, lf_dir='current', only_load="lf")

            optimizer = torch.optim.Adam(lf.parameters(), lr=train_config['lf']['learning_rate'])
            optim_path = os.path.join(train_config['snapshot']['current'], "lf_optim.pt")
            if os.path.exists(optim_path):
                print "Loading Optim Settings"
                optimizer.load_state_dict(safe_load.torch_state(optim_path))
            else:
                print "Failed to load Optim Settings"

        if lowest_loss > sum_loss/steps:
            lowest_loss = sum_loss/steps
            print "Saving Best"

            dirname = train_config['snapshot']['best_validation']
            if not len(dirname) != 0 and os.path.exists(dirname):
                os.makedirs(dirname)

            save_path = os.path.join(dirname, "lf.pt")

            torch.save(lf.state_dict(), save_path)
            lowest_loss_i = 0

        test_loss = sum_loss/steps

        print "Test Loss", sum_loss/steps, lowest_loss
        print "Time:", time.time() - start_time
        print ""

        if allowed_training_time < (time.time() - init_training_time):
            print "Out of time: Exiting..."
            break

        print "Epoch", epoch
        sum_loss = 0.0
        steps = 0.0
        lf.train()
        start_time = time.time()
        for x in train_dataloader:
            if x is None:
                continue
            #Only single batch for now
            x = x[0]
            if x is None:
               continue

            positions = [Variable(x_i.type(dtype), requires_grad=False)[None,...] for x_i in x['lf_xyrs']]
            xy_positions = [Variable(x_i.type(dtype), requires_grad=False)[None,...] for x_i in x['lf_xyxy']]
            img = Variable(x['img'].type(dtype), requires_grad=False)[None,...]

            #There might be a way to handle this case later,
            #but for now we will skip it
            if len(xy_positions) <= 1:
                continue

            reset_interval = 4
            grid_line, _, _, xy_output = lf(img, positions[:1], steps=len(positions), all_positions=positions,
                                               reset_interval=reset_interval, randomize=True, skip_grid=True)

            loss = lf_loss.point_loss(xy_output, xy_positions)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.data[0]
            steps += 1


        print "Train Loss", sum_loss/steps
        print "Real Epoch", train_dataloader.epoch
        print "Time:", time.time() - start_time

    ## Save current snapshots for next iteration
    print "Saving Current"
    dirname = train_config['snapshot']['current']
    if not len(dirname) != 0 and os.path.exists(dirname):
        os.makedirs(dirname)

    save_path = os.path.join(dirname, "lf.pt")
    torch.save(lf.state_dict(), save_path)

    optim_path = os.path.join(dirname, "lf_optim.pt")
    torch.save(optimizer.state_dict(), optim_path)

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
