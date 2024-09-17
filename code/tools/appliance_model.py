import random

import numpy as np
def exponential_fct(x, a, b, c):
    return a + b * np.exp(-c * x)  # for decay model

def logarithmic_fct(x, a, b, c):
    return a + b * np.log(c * x)  # for growth model

# definition of the fundamental appliance model types
class OnOffModel:
    # type = "ON_OFF"
    def __init__(self,on_power, duration):
        self.on = on_power
        self.dur = duration

    def synthesize(self, length):
        return np.linspace(self.on, self.on, length)

class LinearModel:
    # type = "LINEAR"
    def __init__(self, start_power, end_power, duration):
        self.dur = duration
        self.sta = start_power
        self.end = end_power

    def synthesize(self, length):
        return np.linspace(self.sta, self.end, length)

class OnOffGrowthModel:
    # type = "ON_OFF_GROWTH"
    def __init__(self, base_power, scale, stretch, duration):
        self.bas = base_power
        self.sca = scale
        self.stf = stretch
        self.dur = duration

    def synthesize(self, length):
        return logarithmic_fct(np.linspace(1, length, length), self.bas, self.sca, self.stf)

class NoiseModel:
    def __init__(self, low, mean, stdev, high, duration):
        self.dur = duration
        self.low = low
        self.mean = mean
        self.stdev = stdev
        self.high = high

    def synthesize(self, length):
        data = np.random.normal(self.mean, self.stdev, length)
        return np.array([self.bound(x) for x in data])

    def bound(self, val):
        while val > self.high or val < self.low:
            val = np.random.normal(self.mean, self.stdev)
        return val

class RandomRangeModel:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def synthesize(self, length):
        return [random.choice(list(range(int(self.min), int(self.max)))) for i in range(length)]

class CompositeModel:
    def __init__(self, model_num):
        self.model_lst = []

    def add(self, model):
        self.model_lst.append(model)

    def synthesize(self, length_lst):
        synthesized_data = []
        for model, length in zip(self.model_lst, length_lst):
            synthesized_data.append(model.synthesize(length))
        synthesized_data = np.array([i for subsequence in synthesized_data for i in subsequence])
        return synthesized_data