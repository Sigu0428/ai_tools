from ludoObs import LudoObs
import os
import datetime
from matplotlib import pyplot as plt
import numpy as np

class LogEntry():
        def __init__(self, ep, turn, obs, info_dict):
            self.ep = ep
            self.turn = turn
            self.obs = obs
            self.info_dict = info_dict

class Log():    
    def __init__(self, episodes_to_capture=None):
        self.episodes_to_capture = episodes_to_capture
        self.log_entries = []
        self.param_dict = {}
        self.data = {}

    def appendData(self, name, datapoint):
        if name in self.data.keys():
            self.data[name].append(datapoint)
        else:
            self.data[name] = [datapoint]

    def appendInfo(self, ep, turn, obs, info_dict):
        if self.episodes_to_capture is None or any(ep == log_ep for log_ep in self.episodes_to_capture):
            self.log_entries.append(LogEntry(ep, turn, obs, info_dict))

    def dumpToFile(self, method, winrate):
        dir = method + '_runs'
        if not os.path.exists(dir):
            os.makedirs(dir)
        sub_dir = dir + '/{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now()) + '_' + str(winrate)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        else:
            exit()

        with open(sub_dir + '/gamelog.md', 'w') as f:
            prev_ep = -1
            for entry in self.log_entries:
                if entry.ep != prev_ep:
                    prev_ep = entry.ep
                    f.write("## Ep: " + str(entry.ep) + "\n")
                f.write("### Turn:" + str(entry.turn) + " " + LudoObs.getColorOfPlayer(entry.obs.player_i) + "\n")
                f.write(LudoObs.getBoard(entry.obs) + "\n")
                for key, value in entry.info_dict.items():
                    f.write(str(key) + ":" + str(value) + "\n")
                f.write("\n \n")

        for name, data in self.data.items():
            plt.clf()
            plt.plot(data)
            plt.savefig(sub_dir + '/' + name + '.pdf', bbox_inches='tight')
            np.savetxt(sub_dir + '/' + name + '.csv', data, delimiter=",")

        with open(sub_dir + '/params.txt', 'w') as f:
            for key, value in self.param_dict.items():
                f.write(str(key) + ":" + str(value) + "\n")