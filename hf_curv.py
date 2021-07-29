

import glob
import logging
import os
import sys
from logging import Logger, RootLogger
from typing import Any, Union

# import sympy as sp
import numpy as np
import math
import pandas as pd
import scipy.io as sio
from configparser import ConfigParser
import scipy.optimize
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
from matplotlib import pyplot as plt

config = ConfigParser()

import matplotlib.pyplot as plt

logger: Union[Union[Logger, RootLogger], Any] = logging.getLogger('GPSHeading')
logname = 'hf curv.txt'
logging.basicConfig(filename=logname,filemode='w',level=logging.INFO)

config.read('constantsMAT.config', encoding='UTF-8')

# ego vehicle parameters
vehicle_halfWide = 1    # unit - meter
vehicle_long = 2.5    # unit - meter
vehicle_halfWide_po = 1  # unit - meter
vehicle_long_po = 2.5  # unit - meter
vehicle_length = 4  # unit - meter

# Evaluation threshold
ego_speedLimit = 60  # unit - km/h
dy_limit = 3.5  # unit - meter
ay_target = 2  # unit - m/s2
day_p = 3  # unit - m/s3
dx_farLimit = 50  # unit - meter
dx_nearLimit = 20  # unit - meter

def getMatSignal( matStruct, matFile_dict):
    # NOTE you could use the MatFileVO class to read and use your mat file
    matStruct = matStruct.split(";")
    try:
        for i, elem in enumerate(matStruct):
            if i == 0:
                signal = matFile_dict[elem]
            else:
                signal = signal[elem]
    except KeyError:
        signal = None
    return signal

def buildDict(mat, matDict=None):
    if matDict is None:
        matDict = {}
    for key in mat:
        elem = mat[key]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            matDict[key] = {}
            mat2dict(elem, matDict[key])
        elif isinstance(elem, np.ndarray):
            matDict[key] = elem
    return matDict

def mat2dict(mat, matDict=None):
    if matDict is None:
        matDict = {}
    for key in mat._fieldnames:
        elem = mat.__dict__[key]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            matDict[key] = {}
            mat2dict(elem, matDict[key])
        elif isinstance(elem, np.ndarray):
            matDict[key] = elem
    return matDict

def calcDistance(time, vel):
    vel = vel/3.6
    time_diff = list(np.append(np.array(0), np.diff(time)))
    if len(time_diff) > len(vel):
        time_diff.pop()

    distance = np.cumsum(np.multiply(time_diff, vel))
    total_dist = distance[-1]
    total_dist = abs(total_dist)

    return total_dist

def calcCurv(_f):
    line_radius = []
    line_dy = []

    mat = sio.loadmat(mat_speed, struct_as_record=False, squeeze_me=True)
    matspeed = buildDict(mat)
    ts_speed = matspeed["timestamp"]

    mat_01 = sio.loadmat(mat_line01, struct_as_record=False, squeeze_me=True)
    mat01 = buildDict(mat_01)
    mat_02 = sio.loadmat(mat_line02, struct_as_record=False, squeeze_me=True)
    mat02 = buildDict(mat_02)
    ts_line = mat02["timestamp"]

    mat_m = sio.loadmat(mat_map, struct_as_record=False, squeeze_me=True)
    matmap = buildDict(mat_m)
    ts_map = matmap["timestamp"]

    # get information from mat file
    velEgo = getMatSignal("speed", matspeed)
    line01_curv = getMatSignal("Curv", mat01)
    line01_dy = getMatSignal("Dy", mat01)
    line01_prob = getMatSignal("Existprob", mat01)
    line02_curv = getMatSignal("Curv", mat02)
    line02_dy = getMatSignal("Dy", mat02)
    line02_prob = getMatSignal("Existprob", mat02)
    ingeofencing = getMatSignal("geofencing", matmap)

    # vehicle speed
    time = ts_speed[-1]-ts_speed[0]
    dist = calcDistance(ts_speed, velEgo)

    # calculate curv
    r_average_last = 0
    for cyc in range(len(ts_map)):
        cyc_l = 0
        for cycline, tsline in enumerate(ts_line):
            if tsline > ts_map[cyc]:
                cyc_l = cycline - 1
                break
        dev = abs(abs(1/line01_curv[cyc_l]) - abs(1/line02_curv[cyc_l]))
        current_time = cyc/len(ts_line)
        mapstate = 'In geofencing'
        if (line01_prob[cyc_l] > 60 and line02_prob[cyc_l] > 60 and dev < 100 and mapstate == ingeofencing[cyc]):
            a = 1/abs(line01_curv[cyc_l])
            a_1 = abs(line01_curv[cyc_l])
            b = 1/abs(line02_curv[cyc_l])
            b_1 = abs(line02_curv[cyc_l])
            r_average_current = (1/abs(line01_curv[cyc_l]) + 1/abs(line02_curv[cyc_l]))/2
            if r_average_current < 600 and r_average_last > 600:
                print(_f, "%.2f" % (current_time),"%.4f" % line01_curv[cyc_l],"%.4f" % line02_curv[cyc_l])
                r_average_last = r_average_current
            else:
                r_average_last = r_average_current

            line_radius.append(r_average_current)
            linedy = line01_dy[cyc_l]-line02_dy[cyc_l]
            line_dy.append(linedy)
        else:
            pass
    return ts_speed, line_radius, line_dy, dist, time

if __name__ == '__main__':
    dirpath = 'C:/02_Meas/20200115_epl2bp/'
    # Raise warning if trying to run without input file
    if len(sys.argv) != 2:
        logger.warning('Input path is not specified. Default path is used: %s', dirpath)
    else:
        dirpath = sys.argv[1]
        logger.info('Input path: {}'.format(dirpath))
    pathstring = "{}/**"

    # make sure that only folders are processed, and not files in them
    folder_list = [folder for folder in glob.iglob(pathstring.format(dirpath), recursive=True)
                   if os.path.isdir(folder)]

    total_curv = []
    total_dy = []
    total_dist = []
    total_time = []
    total_curv_all = []

    for folder in folder_list:
        # read po target distance
        try:
            mat_speed = os.path.join(folder, 'BCS.mat')
            mat_line01 = os.path.join(folder, 'Line01.mat')
            mat_line02 = os.path.join(folder, 'Line02.mat')
            mat_map = os.path.join(folder, 'MAP.mat')
            if os.path.exists(mat_speed) and os.path.exists(mat_line01) and os.path.exists(mat_line02):
                logger.info('Reading {}.'.format(folder))
            else:
                logger.warning('no data from {}. '
                               'Make sure that it\'s a measurement folder. Skipping.'
                               .format(folder))
                continue

            m_ts_po, curv, dy, dist, time = calcCurv(folder)

            total_curv.extend(curv)
            total_dy.extend(dy)
            total_dist.append(dist)
            total_time.append(time)

        except BaseException as e:
            print(e)
            pass

    # dy chart
    total_dy_all = np.array(total_dy)
    a = len(np.where(total_dy_all < 3)[0])
    b = len(np.where((total_dy_all >= 3) & (total_dy_all < 3.5))[0])
    c = len(np.where(total_dy_all >= 3.5)[0])
    d = len(total_dy_all)
    # print(max(ttc_po_all))
    print(a, b, c, d)

    fig = plt.figure()
    plt.title(os.path.basename(folder))

    labels = ['<3m', '3-3.5m', '>3.5m']
    X = [a, b, c]
    explode = (0.1, 0.02, 0.02)  # 将某一块分割出来，值越大分割出的间隙越大

    fig = plt.figure()
    plt.pie(X, labels=labels, explode=explode, autopct='%1.2f%%', pctdistance=0.8, labeldistance=1.2, startangle=180)
    plt.title("line dy chart")

    fig.savefig(os.path.join(dirpath, 'dy.png'))
    plt.close(fig)

    #curv chart
    total_curv_all = np.array(total_curv)
    a = len(np.where(total_curv_all < 250)[0])
    b = len(np.where((total_curv_all >= 250) & (total_curv_all < 400))[0])
    c = len(np.where((total_curv_all >= 400) & (total_curv_all < 600))[0])
    d = len(np.where((total_curv_all >= 600) & (total_curv_all <= 1500))[0])
    e = len(np.where(total_curv_all > 1500)[0])
    f = len(total_curv_all)

    # print(max(ttc_po_all))
    print(a, b, c, d, e, f)

    fig = plt.figure()
    plt.title(os.path.basename(folder))

    labels = ['<250m', '250-400m', '400-600m', '600-1500m', '>1500m']
    X = [a, b, c, d, e]
    explode = (0.1, 0.1, 0.02, 0.02, 0.02)  # 将某一块分割出来，值越大分割出的间隙越大

    fig = plt.figure()
    plt.pie(X, labels=labels, explode=explode, autopct='%1.2f%%',startangle=180, pctdistance=0.8, labeldistance=1.2)
    plt.title("curve radius chart")

    fig.savefig(os.path.join(dirpath, 'curve_radiusc.png'))
    plt.close(fig)

    # print("---------------------------------------------------")
    print("total distance:", "%.2f" % (np.sum(total_dist)/1000), "km")
    print("total time:", "%.2f" % (np.sum(total_time)/3600), "hours")
    # print("high speed time (V>60kph):", "%.4f" % (np.sum(near_time_total_all)/3600), "hours")
    # print("collision risky time:", "%.4f" % (np.sum(near_time_all)/3600), "hours")
    # print("---------------------------------------------------")
    # print("high speed time in total time:", "%.2f" % (np.sum(near_time_total_all)/np.sum(total_time_all)*100), "%")
    # print("collision risky time percent in total time:", "%.2f" % (np.sum(near_time_all) / np.sum(total_time_all) * 100), "%")
    # print("collision risky time percent in high speed time:", "%.2f" % (np.sum(near_time_all)/np.sum(near_time_total_all)*100), "%")
    # print("---------------------------------------------------")
    # print("average TTC po2 (day=", day_p, "m/s3):", "%.2f" % (np.nanmean(ttc_po2_all)), "second")
    # print("average TTC po3 (day=", day_p, "m/s3):", "%.2f" % (np.nanmean(ttc_po3_all)), "second")
    # print("TTC < 1.5", "%.2f" % ((a+b)/f*100), "%")
    # print("TTC = 1.5-2", "%.2f" % (c/f*100), "%")
    # print("TTC = 2-3", "%.2f" % (d/f*100), "%")
    # print("TTC > 3", "%.2f" % (e/f*100), "%")
    # fig = plt.figure()
    # plt.title(os.path.basename(folder))
    # plt.plot(range(len(ttc_po2_all)), ttc_po2_all, label='ttc po2')
    # plt.gca().set_ylim(0, 1)
    # plt.legend(loc='upper right', prop={'size': 8})
    # fig.savefig(os.path.join(dirpath, os.path.basename(folder) + '_PO2_ttc.png'))
    # plt.close(fig)





