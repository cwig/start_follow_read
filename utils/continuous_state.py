
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn

import sol
from sol.start_of_line_finder import StartOfLineFinder
from lf.line_follower import LineFollower
from hw import cnn_lstm

from utils import safe_load

import numpy as np
import cv2
import json
import sys
import os
import time
import random

def init_model(config, sol_dir='best_validation', lf_dir='best_validation', hw_dir='best_validation', only_load=None):
    base_0 = config['network']['sol']['base0']
    base_1 = config['network']['sol']['base1']

    sol = None
    lf = None
    hw = None

    if only_load is None or only_load == 'sol' or 'sol' in only_load:
        sol = StartOfLineFinder(base_0, base_1)
        sol_state = safe_load.torch_state(os.path.join(config['training']['snapshot'][sol_dir], "sol.pt"))
        sol.load_state_dict(sol_state)
        sol.cuda()

    if only_load is None or only_load == 'lf' or 'lf' in only_load:
        lf = LineFollower(config['network']['hw']['input_height'])
        lf_state = safe_load.torch_state(os.path.join(config['training']['snapshot'][lf_dir], "lf.pt"))
        lf.load_state_dict(lf_state)
        lf.cuda()

    if only_load is None or only_load == 'hw' or 'hw' in only_load:
        hw = cnn_lstm.create_model(config['network']['hw'])
        hw_state = safe_load.torch_state(os.path.join(config['training']['snapshot'][hw_dir], "hw.pt"))
        hw.load_state_dict(hw_state)
        hw.cuda()

    return sol, lf, hw
