import numpy as np

class LMStats(object):
    def __init__(self, default_weight=None):
        if default_weight is None:
            default_weight = -1*np.log(5.0e-16)
        self.default_weight = default_weight

        self.reset()

    def reset(self):
        self.count = 0
        self.accumulated_probablites = None

    def add_stats(self, data):

        #it is probably better to accept softmax data not log_softmax
        softmax_data = np.exp(data)

        #softmax data is shape Timesteps X Characters
        self.count += data.shape[0]

        if self.accumulated_probablites is None:
            self.accumulated_probablites = softmax_data.sum(axis=0)
        else:
            self.accumulated_probablites += softmax_data.sum(axis=0)

    def get_state(self):
        char_log_prob = np.log(self.accumulated_probablites / self.count)
        return LMStatsState(char_log_prob)

class LMStatsState(object):
    def __init__(self, char_log_prob):
        self.char_log_prob = char_log_prob

    def reweight(self, data, alphaweight):
        return data - alphaweight * self.char_log_prob 
