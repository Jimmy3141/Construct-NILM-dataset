import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from scipy import signal

# from .importance_characterization import pid_control
# from .importance_aware_randomization import perturb
import nilmtk
# from .pattern_aware_sampling import sample_points as sap
# from .dynamic_time_wrapping import dtw
import yaml
import shutil
# from .autoLabel import autoLabel
import time
from collections import Counter
class Time:
    def __init__(self, start, end):
        assert start < end, "Start time must be ahead of end time"
        self.start = start
        self.end = end

class ApplianceBase(object):
    def __init__(self, data, type):
        self.df = data
        self.type = type

class ApplianceClass(object):
    def __init__(self, data, type, start, end, sample_rate, used_time=None, number_of_states=None):
        self.df = data
        self.type = self._test_type(type)
        self.start = start
        self.end = end
        self.sample_rate = sample_rate
        self.used_time = []
        self.annotated_data = np.array([])
        self.label_list = []
        self.label_dict = {}
        self.power_df = None

    def plot_appliance(self, zoom = False):
        if zoom:
            annotated_df = self.df[zoom[0]:zoom[1]].copy()
        else:
            annotated_df = self.df.copy()
        plt.plot(annotated_df)
        plt.xlabel("Time")
        plt.ylabel("Power (Watt)")
        plt.title("Consumption of {}".format(self.type.capitalize()))
        for time in self.used_time:
            start,end = time
            start_df = self.df[pd.to_datetime(start):pd.to_datetime(start)+pd.Timedelta(self.sample_rate, unit = "s")]
            end_df = self.df[pd.to_datetime(end):pd.to_datetime(end)+pd.Timedelta(self.sample_rate, unit = "s")]
            start_index = start_df.index[0]
            end_index = end_df.index[0]
            # if start_index in annotated_df.index and end_index in annotated_df.index:
            #     start_value = start_df.values[0]
            #     end_value = end_df.values[0]
            #     plt.plot(start_index, start_value, 'ro--', markersize=4, color="red")
            #     plt.plot(end_index, end_value, 'ro--', markersize=4, color="green")
        plt.xticks(rotation=45)
        plt.legend()

    def label_annotated_data(self, window = 20):
        annotated_df = self.df.copy()
        label_lst = []
        for event_time in self.used_time:
            start,end = event_time
            slice = annotated_df[start:end]
            if len(slice)>window:
                for i in range(0, len(slice), window):
                    labelled_data = slice.values[i:i+window]
                    if len(labelled_data)==window:
                        plt.plot(labelled_data)
                        plt.show()
                        time.sleep(1)
                        if len(self.annotated_data) == 0:
                            self.annotated_data = labelled_data
                        else:
                            self.annotated_data = np.concatenate([self.annotated_data, labelled_data])
                        label = int(input("Do you think the appliance is working? type 0 if it is off"))
                        while label not in [0, 1]:
                            label = int(input("Do you think the appliance is working? type 0 if it is off"))
                        label_lst.append(label)
        self.label_list  = label_lst


    def perturb(self, epsilon, theta, w, k_p=0.7, k_i=0, k_d=0.3):
        time = list(range(1, len(self.df) + 1))
        data = self.df.values
        data = data.reshape(-1)
        pi = 3
        mu = 1
        # 源数据集的读取
        # time, data = rf.read_files(name)
        # LiftVibration的读取
        # sample_points : 采样点集
        sample_points = sap(data, time, theta)
        perturb_array = []
        error_array = [0, 0]
        score_array = [0.01, 0.01]
        alpha = 0.5
        epsilon_remaind = epsilon
        # 对每个采样点进行以下操作：
        for index in range(len(sample_points)):
            if index <= 1:
                perturb_array.append(data[sample_points[index]])
            else:
                # 先在importance_characterization中获取存储了每个采样点的重要度的score array以及每个采样点的误差值的error_array
                score_array, error_array = pid_control(index, data, time, error_array, sample_points, k_p, k_i,
                                                          k_d, pi, score_array)
                # 返回下一个采样点的α，当前采样点的扰动后数值，ε'
                alpha, perturb_result, epsilon_remind = \
                    perturb(data, index, alpha, w, theta, mu, score_array, epsilon, epsilon_remaind,
                                sample_points)
                perturb_array.append(perturb_result)
                # 如果已经过了一个窗口的长度，那么要重置alpha和ε
                if index % w == 0:
                    alpha = 0.5
                    epsilon_remaind = epsilon
                # if index % 100 == 0:
                #     print("100 done")
        process_array = np.round(np.array([data[index] for index in sample_points]))
        perturb_array = np.round(np.array(perturb_array))
        distance = dtw(process_array, perturb_array)
        print("the dtw between perturbed data and original data is{}".format(distance))
        print(sample_points)
        for index, value in zip(sample_points, perturb_array):
            self.df.values[index] = value

    def annotate(self, thres=5, preEventTime=2.0, postEventTime=2.5, votingTime=2.0, minDistance=2.0, m=0.005):
        sr = 1
        power = self.df.values.reshape(-1)
        events, labels = autoLabel(
            power, sr,
            thres, preEventTime=preEventTime, postEventTime=postEventTime, votingTime=votingTime,
            minDistance=minDistance, m=m,
            verbose=True)
        print(len(events), "events")
        # label_counter = Counter(labels).most_common()
        # rare_label = [label for label in label_counter if label[1]<=2]
        # labels = ["S3" if label in rare_label else label for label in labels]
        # print(labels)
        #annotate working power
        labels = [i if i == "S0" else "S1" for i in labels]
        same_index = []
        for index, label in enumerate(labels):
            if index == len(labels) - 2:  # the last but one element
                break
            else:
                if labels[index + 1] == labels[index]:
                    same_index.append(index + 1)
        events = [events[index] for index, value in enumerate(labels) if index not in same_index]
        labels = [value for index, value in enumerate(labels) if index not in same_index]
        #get appliance used time
        label_index_array = [[index1, index1 + 1] for index1, event in enumerate(labels) if event != "S0"]
        if label_index_array[-1][1] > len(events)-1:
            # prevent the last index out of the range
            label_index_array = label_index_array[:-1]
        event_index = [(events[index[0]], events[index[1]]) for index in label_index_array]
        self.used_time = [(self.df.index[i[0]], self.df.index[i[1]])  for i in event_index]

    def label_with_classification(self,window=50, min_class=2, thres=10.0, preEventTime=2.0, postEventTime=2.5, votingTime=2.0, minDistance=2.0, m=0.005):
        sr = 1
        power = self.df.values.reshape(-1)
        power = power.astype(int)
        events, labels = autoLabel(
            power, sr,
            thres, preEventTime=preEventTime, postEventTime=postEventTime, votingTime=votingTime,
            minDistance=minDistance, m=m,
            verbose=True)
        original_events = events
        original_labels = labels.copy()
        print(len(events), "events")
        label_counter = Counter(labels).most_common()
        print(label_counter)
        rare_label = [label[0] for label in label_counter if label[1]<=min_class] # the class of less data than others'
        common_label_lst = [label[0] for label in label_counter if label[0] not in rare_label]
        labels = ["S-1" if label in rare_label else label for label in labels]
        for label in list(set(labels)):
            self.label_dict[label] = []
        #evenet list record the star and end time of an event
        event_lst = []
        for index, value in enumerate(events):
            if index == len(events)-1: #the last but one event
                event_lst.append((event_lst[-1][1],len(power)-1))
                break
            event_lst.append((value, events[index+1]))
        common_label_dic = {}
        for common_label in common_label_lst:
            common_label_dic[common_label] = []
        #common_label_dic key is the label, value is the time series
        for event, label in zip(event_lst, labels):
            if label in common_label_lst:
                common_label_dic[label].append(power[event[0]:event[1]])
        #check the state of common labelled data
        common_states = {}
        print("-"*10)
        print("first round")
        print("-"*10)
        for label, value in common_label_dic.items():
            # key is the label, value is the state
            states = []
            for i in range(min_class): #iterate
                print("Class {}".format(label))
                plt.plot(value[i])
                plt.show()
                time.sleep(1)
                state = int(input("The state of appliance: 0: off, 1: on, 2: abnormal, 3: start, 4: end"))
                while state not in [0,1,2,3,4]:
                    plt.plot(value[i])
                    plt.show()
                    time.sleep(1)
                    state = int(input("The state of appliance: 0: off, 1: on, 2: abnormal"))
                states.append(state)
            max_state = Counter(states).most_common()[0][0]
            common_states[label] = max_state
        labels = [common_states[label1] if label1 in common_label_lst else label1 for label1 in labels]
        index = 0 #the pointer to check current label
        for event, label in zip(event_lst, labels):
            if label=="S-1":
                plt.plot(power[event[0]:event[1]])
                plt.show()
                time.sleep(1)
                state = int(input("The state of appliance: 0: off, 1: on, 2: abnormal, 3:start, 4: end"))
                print("state is {}".format(state))
                time.sleep(0.4)
                while state not in [0,1,2,3,4]:
                    plt.plot(self.label_dict[i], color = "black")
                    plt.show()
                    time.sleep(1)
                    state = int(input("The state of appliance: 0: off, 1: on, 2: abnormal, 3:start, 4: end"))
                labels[index] = state
                index+=1
            else:
                index+=1
        self.label_df = pd.DataFrame(data=[labels, event_lst])
        power_df = pd.DataFrame(power[:-(len(power) % window)].reshape(-1, window))
        power_df["label"] = 5
        label_iter = iter(labels)
        # generate labeled time series with window
        for label, event_interval in zip(labels, event_lst):
            left_pointer = self._find_min_multiple(event_interval[0], window)
            right_pointer = self._find_max_multiple(event_interval[1], window)
            current_label = next(label_iter)
            if left_pointer == right_pointer:
                continue
            for i in range(left_pointer, right_pointer + 1, window):
                power_df.loc[i / window, "label"] = current_label
        power_df = pd.DataFrame(power[:-(len(power) % window)].reshape(-1, window))
        power_df["label"] = 5
        label_iter = iter(labels)
        for label, event_interval in zip(labels, event_lst):
            left_pointer = self._find_min_multiple(event_interval[0], window)
            right_pointer = self._find_max_multiple(event_interval[1], window)
            current_label = next(label_iter)
            if left_pointer == right_pointer:
                continue
            for i in range(left_pointer, right_pointer + 1, window):
                power_df.loc[i / window, "label"] = current_label
        on_or_off_index = power_df[power_df.label == 5].index
        on_or_off_df = power_df.iloc[on_or_off_index, 0:window]
        power_df = power_df.copy()
        for index, row in on_or_off_df.iterrows():
            plt.plot(row)
            plt.show()
            time.sleep(1)
            state = int(input("The state of appliance: 0: off, 1: on, 2: abnormal, 3:start, 4: end"))
            power_df.loc[index, "label"] = state
        self.power_df = power_df

    def _find_min_multiple(self, x, window):
        i = 0
        while True:
            multiple = window * i
            if multiple >= x:
                return multiple
            i += 1

    def _find_max_multiple(self, x, window):
        max_multiple = x - (x % window)
        return max_multiple

    def _test_type(self, type):
        """test if the appliance type is valid"""
        support_type = ['computer monitor', 'dish washer', 'fridge', 'hair dryer', 'light', 'microwave', 'oven', 'rice cooker', 'router', 'running machine', 'set top box',
                        'speaker', 'tablet computer charger', 'television', 'washing machine',"kettle","toaster"]
        if type.lower() in support_type:
            return type.lower()
        else:
            raise Exception("Do not support this type of appliance. Accepted appliances are {}".format(support_type))

    def labeled_data_to_csv(self):
        pth = r"../save/{}/{}_to_{}".format(str(self.type)+str(self.builidng)+str(self.meter),str(self.start).replace(" ", "").replace(":", "_"), str(self.end).replace(" ", "").replace(":", "_"))
        if not os.path.exists(pth):
            os.makedirs(pth)
        self.power_df = self.power_df.dropna()
        self.power_df.to_csv(os.path.join(pth, "labeled_data.csv"))

class ApplianceFromH5(ApplianceClass):
    def __init__(self, h5_path, type, building, meter, start, end, sample_rate=6, data_type = "active"):
        self.data_location = h5_path
        self.building = building
        self.sample_rate = sample_rate
        self.meter = meter
        self.start = start
        self.end = end
        self.data_type = data_type
        df = self._get_power()
        super().__init__(df, type, start, end, sample_rate)

    def replace_with_generative_data(self, same_building=True):
        appliance_path = os.path.join(r"../data/timegan_data", self.type)
        if same_building:
            meter = "b" + str(self.building) + "m" + str(self.meter)
            meter_path = os.path.join(appliance_path ,meter)
        else:
            meter_path = os.path.join(appliance_path, random.choice(appliance_path))
        for time in self.used_time:
            time_period = len(self.df[time[0]:time[1]])
            ratio = int(self.sample_rate/6)
            np_array = self._get_nparray(meter_path, time_period, ratio)
            self.df[time[0]:time[1]] = np_array

    def _get_nparray(self, path, length, ratio):
        npy_lst = os.listdir(path)
        random.shuffle(npy_lst)
        appliance = np.array([])
        appliances_path = [os.path.join(path, npy_path) for npy_path in npy_lst]
        while len(appliance) <length:
            appliance_data = np.load(random.choice(appliances_path))
            if len(appliance_data)%ratio != 0:
                left = len(appliance_data) % ratio
                appliance_data = appliance_data[left:]
            appliance = np.concatenate([appliance, appliance_data.reshape(-1, ratio).mean(axis=1)])
        return appliance[:length].reshape(length, -1)

    def _get_power(self):
        hdf = pd.HDFStore(self.data_location)
        with open(self.data_location) as f:
            dir = "building{}/elec/meter{}".format(self.building, self.meter)
            power = hdf.get(dir)
            power.index.name = 'Date'
            power.reset_index(inplace=True)
            power['Date'] = pd.to_datetime(power['Date']).dt.tz_localize(None)
            # 转化时间戳
            power.set_index('Date', drop=True, inplace=True)
            power = power.sort_index()
            power = power["power"][self.data_type].to_frame()  # get active
            power = power[self.start:self.end].astype(int)
            sampled_power = power.resample("{}S".format(self.sample_rate))
            non_null_power = sampled_power.fillna(0)
            non_null_power[non_null_power<0] = 0
        return non_null_power

    def labeled_data_to_csv(self):
        pth = r"../save/{}/{}_to_{}".format(str(self.type)+"/b"+str(self.building)+"m"+str(self.meter)+"/",str(self.start).replace(" ", "").replace(":", "_"), str(self.end).replace(" ", "").replace(":", "_"))
        if not os.path.exists(pth):
            os.makedirs(pth)
        self.power_df = self.power_df.dropna()
        self.power_df.to_csv(os.path.join(pth, "labeled_data.csv"))

class GeneratedAppliance(ApplianceClass):
    def __init__(self, type, time, sample_rate, used_time, number_of_states=None, dataset_type="labeled_data", mean=None):
        """ApplianceData store the dataframe of an appliance and enable data transformation"""
        self.start = time.start
        self.end = time.end
        if mean != None:
            self.mean = int(mean)
        self.dataset_type = dataset_type
        df = self._create_df(type, self.start, self.end, sample_rate, used_time, dataset_type, mean)
        super().__init__(df, type, time.start, time.end, sample_rate,number_of_states=number_of_states)
        self.used_time = used_time #used time must be an iterable of iterables following the rules [(start1, end1), (start2, end2)]

    def reschedule(self, time_period):
        """time period must conform to the format [(start, end)]"""
        pass

    def smooth(self):
        pass

    def regenerate(self):
        pass

    def delete_and_interpolate(self):
        pass

    def _create_df(self, type, start, end, sample_rate, used_time, dataset_type="labeled_data", mean=None):
        self._test_time_valid(start, end, used_time)
        # create dataframe for the appliance
        index = pd.date_range(start=start, end=end, freq=str(sample_rate) + "S")
        df = pd.DataFrame(index=index, data=[0] * len(index))#df is the created dataframe
        df.index.name = "Date"
        df.columns = ["active"]
        # insert the required time into gaps in df
        for appliance_start, appliance_end in used_time:
            sample_len = len(df[appliance_start:appliance_end])
            sampled_raw_data = self._retrive_sampled_raw_data(type,  sample_len, sample_rate, dataset_type, mean)
            df[appliance_start:appliance_end] = sampled_raw_data  # fill in the time gaps
        if mean != None:
            df = self._set_mean(df, mean)  # rescale the dataframe to required mean
        return df

    def _retrive_sampled_raw_data(self, type, length, sample_rate, dataset_type, mean):
        _test_dataset_type(dataset_type)  # choose dataset type as labeled_data or timegan_data
        # import raw data
        raw_data = self._return_raw(type, dataset_type)
        while not len(raw_data)>length:
            raw_data_new = self._return_raw(type, dataset_type)
            raw_data = np.concatenate((raw_data_new, raw_data))
        if sample_rate != 8: # the original sample rate is 8
            raw_data = self._retrive_sampled_raw_data(type, sample_rate, 8, dataset_type, mean)#resample the raw_data to required rate
        # sample points from raw data
        process_data = self._sample_points(raw_data, length)
        return process_data

    def _return_raw(self, type, dataset_type):
        # retrive a randomly chosen data file
        raw_data_root = "..//data//" + dataset_type + "//" + str(type)
        random_dataset = random.choice(os.listdir(raw_data_root))
        if dataset_type == "labeled_data":
            raw_dataset_path = raw_data_root + "//" + random_dataset
            random_data_path = random.choice(raw_dataset_path)
            while not random_data_path.endswith("csv"):
                raw_dataset_path = raw_data_root + "//" + random_dataset
                random_data_path = random.choice(os.listdir(raw_dataset_path))
            raw_data = pd.read_csv(raw_dataset_path+"//"+random_data_path)  # all the labeled raw data
            # raw_data.set_index(raw_data.Date, inplace=True)
            # raw_data = raw_data.iloc[:,1:]
            return raw_data.iloc[:,1:].values
        elif dataset_type == "timegan_data":
            raw_data_path = raw_data_root + "//" + random_path + "//all.npy"
            raw_data = np.load(raw_data_path)
            return raw_data

    def _sample_points(self, raw_data, length):
        total_len = len(raw_data)
        start_point = 0
        process_data = raw_data[start_point:start_point + length]
        return process_data

    def _set_mean(self, df, mean):
        original_mean = df[df > 5].mean()  # most of the standby power is lower than 5 W
        scale = mean / original_mean
        df_new = df[df > mean] * scale
        df_new = df_new.fillna(0)
        return df_new

    def _test_time_valid(self, start, end, used_time):
        assert start <= end, "Start time must be ahead of end time"
        # for time_period in used_time:
        #     assert len(time_period) == 2, "{} doesn't reform to the format (start, end)".format(time_period)
        #     period_start, period_end = time_period
        #     assert period_start < period_end, "start time is {}, end time is {}, start time must be ahead of end time".format(
        #         period_start, period_end)


class GeneratedAppliance(ApplianceClass):
    def __init__(self, type, time, sample_rate, used_time, number_of_states=None, dataset_type="labeled_data", mean=None):
        """ApplianceData store the dataframe of an appliance and enable data transformation"""
        self.start = time.start
        self.end = time.end
        if mean != None:
            self.mean = int(mean)
        self.dataset_type = dataset_type
        df = self._create_df(type, self.start, self.end, sample_rate, used_time, dataset_type, mean)
        super().__init__(df, type, time.start, time.end, sample_rate,number_of_states=number_of_states)
        self.used_time = used_time #used time must be an iterable of iterables following the rules [(start1, end1), (start2, end2)]

    def reschedule(self, time_period):
        """time period must conform to the format [(start, end)]"""
        pass

    def smooth(self):
        pass

    def regenerate(self):
        pass

    def delete_and_interpolate(self):
        pass

    def _create_df(self, type, start, end, sample_rate, used_time, dataset_type="labeled_data", mean=None):
        self._test_time_valid(start, end, used_time)
        # create dataframe for the appliance
        index = pd.date_range(start=start, end=end, freq=str(sample_rate) + "S")
        df = pd.DataFrame(index=index, data=[0] * len(index))#df is the created dataframe
        df.index.name = "Date"
        df.columns = ["active"]
        # insert the required time into gaps in df
        for appliance_start, appliance_end in used_time:
            sample_len = len(df[appliance_start:appliance_end])
            sampled_raw_data = self._retrive_sampled_raw_data(type,  sample_len, sample_rate, dataset_type, mean)
            df[appliance_start:appliance_end] = sampled_raw_data  # fill in the time gaps
        if mean != None:
            df = self._set_mean(df, mean)  # rescale the dataframe to required mean
        return df

    def _retrive_sampled_raw_data(self, type, length, sample_rate, dataset_type, mean):
        _test_dataset_type(dataset_type)  # choose dataset type as labeled_data or timegan_data
        # import raw data
        raw_data = self._return_raw(type, dataset_type)
        while not len(raw_data)>length:
            raw_data_new = self._return_raw(type, dataset_type)
            raw_data = np.concatenate((raw_data_new, raw_data))
        if sample_rate != 8: # the original sample rate is 8
            raw_data = self._retrive_sampled_raw_data(type, sample_rate, 8, dataset_type, mean)#resample the raw_data to required rate
        # sample points from raw data
        process_data = self._sample_points(raw_data, length)
        return process_data

    def _return_raw(self, type, dataset_type):
        # retrive a randomly chosen data file
        raw_data_root = "..//data//" + dataset_type + "//" + str(type)
        random_dataset = random.choice(os.listdir(raw_data_root))
        if dataset_type == "labeled_data":
            raw_dataset_path = raw_data_root + "//" + random_dataset
            random_data_path = random.choice(raw_dataset_path)
            while not random_data_path.endswith("csv"):
                raw_dataset_path = raw_data_root + "//" + random_dataset
                random_data_path = random.choice(os.listdir(raw_dataset_path))
            raw_data = pd.read_csv(raw_dataset_path+"//"+random_data_path)  # all the labeled raw data
            # raw_data.set_index(raw_data.Date, inplace=True)
            # raw_data = raw_data.iloc[:,1:]
            return raw_data.iloc[:,1:].values
        elif dataset_type == "timegan_data":
            raw_data_path = raw_data_root + "//" + random_path + "//all.npy"
            raw_data = np.load(raw_data_path)
            return raw_data

    def _sample_points(self, raw_data, length):
        total_len = len(raw_data)
        start_point = 0
        process_data = raw_data[start_point:start_point + length]
        return process_data

    def _set_mean(self, df, mean):
        original_mean = df[df > 5].mean()  # most of the standby power is lower than 5 W
        scale = mean / original_mean
        df_new = df[df > mean] * scale
        df_new = df_new.fillna(0)
        return df_new

    def _test_time_valid(self, start, end, used_time):
        assert start <= end, "Start time must be ahead of end time"
        # for time_period in used_time:
        #     assert len(time_period) == 2, "{} doesn't reform to the format (start, end)".format(time_period)
        #     period_start, period_end = time_period
        #     assert period_start < period_end, "start time is {}, end time is {}, start time must be ahead of end time".format(
        #         period_start, period_end)


class GeneratedRefitAppliance():
    def __init__(self, type, time, sample_rate, used_time, data_rate, number_of_states=None):
        """ApplianceData store the dataframe of an appliance and enable data transformation"""
        self.start = time.start
        self.sample_rate = sample_rate
        self.end = time.end
        self.type =type
        self.true_activate_time = []
        self.number_of_states = number_of_states
        self.sample_data_path = {}
        self.used_time = used_time #used time must be an iterable of iterables following the rules [(start1, end1), (start2, end2)
        df = self._create_df(type, self.start, self.end, sample_rate, used_time, data_rate)
        self.df = df

    def plot_appliance(self, zoom = False):
        self.df.plot()

    def _create_df(self, type, start, end, sample_rate,used_time, rate):
        self._test_time_valid(start, end, used_time)
        # create dataframe for the appliance
        index = pd.date_range(start=start, end=end, freq=str(sample_rate) + "S")
        df = pd.DataFrame(index=index, data=[0] * len(index))#df is the created dataframe
        df.index.name = "Date"
        df.columns = ["active"]
        # insert the required time into gaps in df
        data_path = r"../data/refit_data/b2/" + str(rate) + "/" + self.type+ "/"
        samples_data_path = [data_path + i for i in os.listdir(data_path) if i.endswith(".csv")]
        samples_data_path = np.random.choice(samples_data_path, len(self.used_time))
        delta = datetime.timedelta(0,self.sample_rate)
        for i, time in enumerate(used_time):
            appliance_start = time[0]
            sampled_raw_data = pd.read_csv(samples_data_path[i]).iloc[:,1:].values
            appliance_end_time=str(datetime.datetime.strptime(appliance_start,"%Y-%m-%d %H:%M:%S")+delta*(len(sampled_raw_data)-1))
            df[appliance_start:appliance_end_time] = sampled_raw_data  # fill in the time gaps
            self.true_activate_time.append((appliance_start, appliance_end_time))
            self.sample_data_path[(appliance_start, appliance_end_time)] = samples_data_path[i]
        return df


    def _test_time_valid(self, start, end, used_time):
        assert start <= end, "Start time must be ahead of end time"
        # for time_period in used_time:
        #     assert len(time_period) == 2, "{} doesn't reform to the format (start, end)".format(time_period)
        #     period_start, period_end = time_period
        #     assert period_start < period_end, "start time is {}, end time is {}, start time must be ahead of end time".format(
        #         period_start, period_end)

class Appliance_from_csv(ApplianceClass):
    def __init__(self, type,start,end,sample_rate, csv_path="",used_time=None, number_of_states=None):
        self.type = type
        self.end = end
        self.sample_rate = sample_rate
        self.csv_path = csv_path
        df=pd.read_csv(csv_path)
        df.index = pd.to_datetime(df["Date"])
        df = df.iloc[:,1:]
        self.df =df

class Building:
    def __init__(self, id, time, sample_rate):
        self.id = self._test_id(id)
        self.start, self.end = time.start, time.end
        self.index = pd.date_range(start=self.start, end=self.end, freq=str(sample_rate)+"S")
        self.aggregated = pd.DataFrame(index=self.index, data=[0]*len(self.index))
        self.aggregated.index.name = "Date"
        self.aggregated.columns = ["active"]
        self.appliance_lst = []
        self.meta_data_location = None
        self.data_location = None
        self.sample_rate = sample_rate
        self.state_dict = {"washing machine":4,"television":2,"dishwasher":3, "microwave":2,"toaster":3,"kettle":2,"fridge":2}

    def load_aggregate(self, path, time):
        aggregate = pd.read_csv(path)
        aggregate = aggregate.set_index("Date")
        aggregate.index = pd.date_range(time.start, end=time.end, freq=str(self.sample_rate) + "S")
        aggregate.index.name = "Date"
        self.aggregated = aggregate

    def add_appliance(self, appliance):
        self.appliance_lst.append(appliance)
        self.aggregated += appliance.df

    def plot_aggregated_data(self):
        plt.plot(self.aggregated)

    def oversample(self, over_sample_time, appliance, rate): # rate is the required amout of data
        df = [i.df for i in self.appliance_lst if i.type ==appliance][0]
        data_path = r"../data/refit_data/b2/" + str(rate) + "/" + appliance + "/"
        samples_data_path = [data_path + i for i in os.listdir(data_path) if i.endswith(".csv")]
        samples_data_path = np.random.choice(samples_data_path, len(over_sample_time))
        delta = datetime.timedelta(0, self.sample_rate)
        for i, time in enumerate(over_sample_time):
            appliance_start = time[0]
            if datetime.datetime.strptime(self.start,"%Y-%m-%d %H:%M:%S")<appliance_start < datetime.datetime.strptime(self.end,"%Y-%m-%d %H:%M:%S"):
                appliance_start = df[appliance_start:appliance_start+datetime.timedelta(0, int(self.sample_rate))].index[0]
                sampled_raw_data = pd.read_csv(samples_data_path[i]).iloc[:, 1:].values
                appliance_end_time = str(
                       appliance_start + delta * (len(sampled_raw_data)-1))
                if datetime.datetime.strptime(
                        appliance_end_time, "%Y-%m-%d %H:%M:%S") < datetime.datetime.strptime(
                        self.end, "%Y-%m-%d %H:%M:%S"):
                    df[appliance_start:appliance_end_time] += sampled_raw_data  # fill in the time gaps #取出索引修改
                    self.aggregated[appliance_start:appliance_end_time] += sampled_raw_data

    def to_dataset(self, path=None):
        """building implies the name of the folder created, for example building_1."""
        if path == None:
            path = "..//dataset//house_"+str(self.id)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        self.data_location = path
        self.aggregated.to_csv(path+"//channel_1.csv")
        # convert aggregated data to csv
        for id, appliance in enumerate(self.appliance_lst):
            appliance.df.to_csv(path+"//channel_"+str(id+2)+".csv")
            #convert appliance data to csv

    def create_metadata(self):
        #create metadata for the building.
        self._create_building_yaml()
        self._create_dataset_yaml()
        self._create_meter_devices()

    def _create_building_yaml(self):
        #create building.yaml file.
        #1. for center meter
        building = {
            "instance":self.id,
            "original_name": 'house'+str(self.id),
            "elec_meters":{
                1: {'site_meter': True,
                    'device_model': 'eMonitor',
                    'data_location': '/building1/elec/meter1'
                }
            }
        }

        #2. for meters
        for i, appliance in enumerate(self.appliance_lst):
            building["elec_meters"][i+2] = {
                'submeter_of': 0,
                'device_model': 'eMonitor',
                'data_location': '/building1/elec/meter{}'.format(str(i+2))
            }
        building["appliances"]=[]
        #3. for appliances
        for i, appliance in enumerate(self.appliance_lst):
            new_appliance = {
                'original_name':appliance.type,
                'type':appliance.type,
                'instance':1,
                'meters':[]
            }
            new_appliance["meters"].append(i+2)
            building["appliances"].append(new_appliance)

        root = '..//dataset//metadata//building{}'.format(self.id)
        self.meta_data_location = root
        if os.path.exists(root):
            shutil.rmtree(root)
        os.makedirs(root)
        3.#load data into yaml
        with open(root+'//building{}.yaml'.format(self.id, self.id), 'w', encoding='utf-8') as f:
            yaml.dump(data=building, stream=f, allow_unicode=True)

    def _create_dataset_yaml(self):
        dataset_info = {'name': 'PNILM',
 'long_name': 'A Toolbox for the Evaluation of Privacy Preserving Non-Intrusive Load Monitoring Based on Local Differential Privacy',
 'creators': ['Junxiang Tang', "Huan Yang", "Mingxiang Wang"],
 'publication_date': 2023,
 'institution': 'University of Aberdeen',
 'contact': 'u18jt21abdn.ac.uk',
 'description': 'Self-defined dataset.',
 'subject': 'Disaggregated power demand from domestic buildings.',
 'number_of_buildings': 1,
 'timezone': 'US/Eastern',
 'geo_location': {'locality': 'Aberdeen',
  'country': 'UK',
  'latitude': 42.360091,
  'longitude': -71.09416},
 'related_documents': ['http://redd.csail.mit.edu',
  'J. Zico Kolter and Matthew J. Johnson. REDD: A public data set for energy disaggregation research. In proceedings of the SustKDD workshop on Data Mining Applications in Sustainability, 2011. http://redd.csail.mit.edu/kolter-kddsust11.pdf\n'],
 'schema': 'https://github.com/nilmtk/nilm_metadata/tree/v0.2'}
        #store file
        root = '..//dataset//metadata//building{}'.format(self.id)
        with open(root+'//dataset.yaml'.format(self.id), 'w', encoding='utf-8') as f:
            yaml.dump(data=dataset_info, stream=f, allow_unicode=True)

    def _create_meter_devices(self):
        meter_device = {'eMonitor': {'model': 'eMonitor',
  'manufacturer': 'Powerhouse Dynamics',
  'manufacturer_url': 'http://powerhousedynamics.com',
  'description': 'Measures circuit-level power demand.  Comes with 24 CTs. This FAQ page suggests the eMonitor measures real (active) power: http://www.energycircle.com/node/14103  although the REDD readme.txt says all channels record apparent power.\n',
  'sample_period': 6,
  'max_sample_period': 6,
  'measurements': [{'physical_quantity': 'power',
    'type': 'active',
    'upper_limit': 5000,
    'lower_limit': 0}],
  'wireless': False}}
        root = '..//dataset//metadata//building{}'.format(self.id)
        with open(root+'//meter_devices.yaml'.format(self.id), 'w', encoding='utf-8') as f:
            yaml.dump(data=meter_device, stream=f, allow_unicode=True)


    def _test_id(self, id):
        try:
            id = int(id)
            return id
        except:
            raise Exception("id must be integer", "the input is {}".format(id))

    #sample a continuous dataframe for inserting into the required path


class Building_from_csv(Building):
    def __init__(self, id, time, sample_rate, dataset_path):
        super().__init__(id, time, sample_rate)
        self.dataset_path = dataset_path
        df = self._retrieve_data()

    def _retrieve_data(self):
        df=pd.read_csv(self.dataset_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        aggregated = df["Aggregate"].to_frame()
        aggregated.columns = ["active"]
        self.aggregated = aggregated
        df = df.iloc[:,1:]
        for i in df.columns:
            appliance_data = df.loc[:, [i]]
            appliance_data.columns = ["active"]
            appliance = ApplianceBase(appliance_data, i)
            self.appliance_lst.append(appliance)
            print("*"*10)
            print(i)
            print("*"*10)

def _test_dataset_type(dataset_type):
    available_type = ["labeled_data", "timegan_data"]
    if dataset_type not in available_type:
        raise Exception("Do not support this type of appliance. Accepted dataset types are {}".format(available_type))

if __name__ == '__main__':
    print()



