import os
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from ..plotting.plot import watermark
plt.style.use('kit')


def poly4(x, a=1.0, b=1.0, c=1.0, d=1.0, e=1.0):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def poly2(x, a=1.0, b=0.0, c=0.0):
    return a * x**2 + b * x + c

def poly1(x, a=1.0, b=0.0):
    return a*x + b

def exp(x, a=1.0, b=1.0, c=0.0):
    return a * np.exp(-b * x) + c

def exp2(x, a=20.0, b=-1.5, c=0.0):
    return a * np.exp(-b * x) + c



class hls4ml_scans():
    
    def __init__(self, file, keys):
        
        self.file = file
        self.keys = keys
        self.bb = ['BRAM', 'DSP', 'FF', 'LUT']
        self.scans = self.get_scans()
        
        zcu = {'BRAM': 1824, 'DSP': 2520, 'FF': 548160, 'LUT': 274080}
        highflex = {'BRAM': 1200, 'DSP': 2928, 'FF': 597120, 'LUT': 298560}

    def get_scans(self):
        scans = {}
        for k in self.keys:
            scans[k]= []
            
        with open(self.file, 'r') as f:
            lines = f.readlines()[6:]
            for i, line in enumerate(lines):
                if 'scan for' in line:
                    scan = line.split(' ')
                    key = scan[2]
                    config_dict = json.loads(lines[i+2].strip(' \n').replace("'", "\""))
                    if key == 'n_nodes&n_edges' or key == 'n_nodes&2n_edges':
                        value = config_dict['n_nodes']
                    else:
                        value = config_dict[key]
                    minutes, seconds = lines[i+3].split(' ')[2:4]
                    comp_time = float(minutes.replace('min',''))*60 + int(seconds.replace('s','')) # in seconds
                    utilisation = [int(l.split(' ')[1]) for l in lines[i+7:i+11]]
                    util = {}
                    for ib, b in enumerate(self.bb):
                        util[b] = utilisation[ib]

                    scans[key].append([value, config_dict, comp_time, util])
                                       
#         for k, v in scans.items():
#             setattr(self, k, v)

        return scans
    
    def show_compilation_time(self, key, fit_func=None, *params):
        scans = np.asarray(self.scans[key]).T
        value, config , comp_time, utils = scans       
        if key == 'precision':
            value = [int(v.strip('ap_fixed<').strip('>')[:2].strip(',')) for v in value]
        value = np.array(value, dtype=float)  
        plt.plot(value, comp_time, label='data')
        if fit_func:
            popt, pcov = curve_fit(fit_func, value, comp_time, *params)
            plt.plot(value, fit_func(value, *popt), '--', label=f'{str(fit_func)}')
        plt.ylabel('compilation time (s)')
        plt.xlabel(key)
        watermark(py=0.9, shift=0.2, scale=1.4)
        plt.legend()
        
    def show_occupancy(self, key):

        scans = np.asarray(self.scans[key]).T
        value, config , comp_time, utils = scans  
        if key == 'precision':
            value = [int(v.strip('ap_fixed<').strip('>')[:2].strip(',')) for v in value]
#         plt.hlines(100, min(value), max(value), color='grey', linestyle='--')
        u = {}
        for k in self.bb:
            u[k]= []
        for i, val in enumerate(value):
            for k, v in utils[i].items():
                u[k].append(v) 
        for k, v in u.items():
            plt.plot(value, np.array(v)/self.zcu[k]*100, label=f'{k}')
        watermark(py=0.9, shift=0.2, scale=1.6)
        plt.legend(loc='upper right')
        plt.ylabel('occupancy on ZCU102 (%)')
        plt.xlabel(key)


    def show_occupancies(self):
        plt.figure(figsize=(40,15))
        for i,k in enumerate(self.keys, 1):
            plt.subplot(2,4,i)
            test.show_occupancy(k)
        plt.show()
        
    def show_compilation_times(self):
        plt.figure(figsize=(40,15))
        for i,k in enumerate(self.keys, 1):
            plt.subplot(2,4,i)
            test.show_compilation_time(k)
        plt.show()        
    