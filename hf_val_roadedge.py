

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
logging.basicConfig(level=logging.INFO)

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

def getPOvy(time, po_index, vy_po):
    po_vy = []
    for cyc in range(len(time)):
        if po_index[cyc] <= 31:
            po_vyv = vy_po[po_index[cyc], cyc]
        else:
            po_vyv = 0
        po_vy.append(po_vyv)

    po_vy = np.array(po_vy)
    return po_vy

def getPOvx(time, po_index, vx_po):
    po_vx = []
    for cyc in range(len(time)):
        if po_index[cyc] <= 31:
            po_vxv = vx_po[po_index[cyc], cyc]
        else:
            po_vxv = 0
        po_vx.append(po_vxv)

    po_vx = np.array(po_vx)
    return po_vx

def calcDistance(time, vel):
    vel = vel * 0.0039 - 0.0119
    time_diff = list(np.append(np.array(0), np.diff(time)))
    if len(time_diff) > len(vel):
        time_diff.pop()

    distance = np.cumsum(np.multiply(time_diff, vel))
    total_dist = distance[-1]
    total_dist = abs(total_dist)

    return total_dist


def collisionRisk_long(ttc, dx_po, vx_po):
    dx_po = dx_po * 0.0625
    vx_po = vx_po * 0.0312 - 128
    moving_distance = ttc*vx_po
    distance_po = dx_po + moving_distance
    if (ttc == 0) | (distance_po > (dx_nearLimit + 4)) | (distance_po < (dx_nearLimit - 4)):
        return False
    else:
        return True

def collisionRisk_rd(po_exist, rd_exist, dy_rd):
        if rd_exist > 60 and dy_rd < 3:
            return True
        else:
            return False


# def predict_right_dtlc(self, measurement_time, line_right, line_right_aplPsi, line_right_kapHor, line_right_kapDtHor,
#                        ego_velocity):
#     """
#     predict right DTLC
#     dy = c0 + c1*dx + 1/2*c2sqr(dx) + 1/6*c3cube(dx)
#     """
#     predict_time = self.config['PREDICTION_TIME']
#     dx_right = predict_time * ego_velocity
#     y_right_predict = line_right + line_right_aplPsi * dx_right + 0.5 * line_right_kapHor * pow(dx_right, 2) + 0.166667 * line_right_kapDtHor * pow(dx_right, 3)
#     return y_right_predict

def calcDTC(dy_po):
    dy_po = dy_po * 0.0625 - 64
    if dy_po == 0:
        dtc = 0
    elif dy_po > 0:
        dtc = dy_po - vehicle_halfWide - vehicle_halfWide_po
    else:
        dtc = -dy_po - vehicle_halfWide - vehicle_halfWide_po
    return dtc

def calcDTC_rd(dy_rd):
    if dy_rd == 0:
        dtc = 0
    elif dy_rd > 0:
        dtc = dy_rd - vehicle_halfWide
    else:
        dtc = -dy_rd - vehicle_halfWide
    return dtc


def calcTTC(dtc, ayv):
    coe = 6*dtc/ayv
    if dtc != 0:
        ttc = pow(abs(coe), 1/3)
    else:
        ttc = 0

    return ttc
    # coe = (1/6)*ayv
    # t = sp.Symbol('t')
    # d = coe*t**3 + vy*t
    # t = sp.solve(d)
    # if max(t) > 0:
    #     return max(t)
    # else:
    #     return 0

def calcTTA():
    t = ay_target/day_p

    return t

def ttc_final(dtc,ttc):

    v2 = 0.5*pow(calcTTA(), 2)*day_p
    d_po2_rest = dtc - day_p * pow(calcTTA(), 3) / 6
    if d_po2_rest <= 0:
        ttc_f = ttc
    elif d_po2_rest > 0:
        ttc_f = ttc + calcT2(ay_target / 2, v2, -d_po2_rest)

    return ttc_f

def calcT2(a,b,c):
    delta = pow((b*b - 4*a*c), 1/2)
    if delta >= 0:
        x1 = ((-b + delta)/2*a)
        x2 = ((-b - delta)/2*a)
        if x1 >= x2:
            x = x1
        elif x2 > x1:
            x = x2
    else:
        x = 0

    return x

def calcTTC_left(ay,vy,dy):
    # a: acceleration
    # b: vy
    # d: dy
    a = 0.5*ay
    b = -vy
    c = -dy
    d = b*b - 4*a*c
    if d < 0:
        return 0
    elif d == 0:
        x1 = -b/2*a
        x2 = x1
    else:
        x1 = (-b + math.sqrt(d))/(2*a)
        x2 = (-b - math.sqrt(d))/(2*a)

    if x1 > 0:
        x = x1
    elif x2 > 0:
        x = x2
    else:
        x = 0
    return x

def calcTTC_right(ay,vy,dy):
    # a: acceleration
    # b: vy
    # d: dy
    a = 0.5*ay
    b = vy
    c = dy
    d = b*b - 4*a*c
    if d < 0:
        return 0
    elif d == 0:
        x1 = -b / 2 * a
        x2 = x1
    else:
        x1 = (-b + math.sqrt(d))/(2*a)
        x2 = (-b - math.sqrt(d))/(2*a)

    if x1 > 0:
        x = x1
    elif x2 > 0:
        x = x2
    else:
        x = 0
    return x

def calcDistanceKPI():
    near_array = []
    near_array_total = []
    near_array_po2 = []
    near_array_po3 = []
    ttc_po2_array = []
    ttc_po3_array = []

    mat_po_data = sio.loadmat(mat_po, struct_as_record=False, squeeze_me=True)
    mat_roadedgele = sio.loadmat(mat_roadedge_le, struct_as_record=False, squeeze_me=True)
    mat_roadedgeri = sio.loadmat(mat_roadedge_ri, struct_as_record=False, squeeze_me=True)
    mat_ego_speed = sio.loadmat(mat_speed, struct_as_record=False, squeeze_me=True)
    # po and spped ts are same
    matDictFus_po = buildDict(mat_po_data)
    matDictFus_speed = buildDict(mat_ego_speed)
    ts_po = matDictFus_po["timestamp"]
    #road edge
    matDictFus_roadedge_ri = buildDict(mat_roadedgeri)
    matDictFus_roadedge_le = buildDict(mat_roadedgele)
    ts_roadedge = matDictFus_roadedge_ri["timestamp"]


    # get information from mat file
    velEgo = getMatSignal("ego;speed", matDictFus_speed)
    # po2_index = getMatSignal("PO;i2;index", matDictFus)
    po2_dx = getMatSignal("PO;Le;dx", matDictFus_po)
    po2_dy = getMatSignal("PO;Le;dy", matDictFus_po)
    po2_vx = getMatSignal("PO;Le;vx", matDictFus_po)
    po2_vy = getMatSignal("PO;Le;vy", matDictFus_po)
    po2_existprob = getMatSignal("PO;Le;ExistProb", matDictFus_po)
    # po3_index = getMatSignal("PO;i3;index", matDictFus)
    po3_dx = getMatSignal("PO;Ri;dx", matDictFus_po)
    po3_dy = getMatSignal("PO;Ri;dy", matDictFus_po)
    po3_vx = getMatSignal("PO;Ri;vx", matDictFus_po)
    po3_vy = getMatSignal("PO;Ri;vy", matDictFus_po)
    po3_existprob = getMatSignal("PO;Ri;ExistProb", matDictFus_po)
    # po_vy = getMatSignal("PO;vyv", matDictFus)  # 0-32 row
    # po_vx = getMatSignal("PO;vxv", matDictFus)  # 0-32 row

    # road edge information
    roadedge_ri_dy = getMatSignal("Line;Dy", matDictFus_roadedge_ri)
    roadedge_ri_existprob = getMatSignal("Line;ExistProb", matDictFus_roadedge_ri)

    roadedge_le_dy = getMatSignal("Line;Dy", matDictFus_roadedge_le)
    roadedge_le_existprob = getMatSignal("Line;ExistProb", matDictFus_roadedge_le)
    # --------------------------------------------------------------------------------------------------------------
    # get v > 60 kph (16.67 m/s) and dy < 2.5m
    velEgo_limit = ego_speedLimit /3.6  # (config["ego_vehicle"].getint("ego_speedLimit"))

    # test
    # for cyc in range(len(ts_po)):
    #     # calculate dlc
    #     dlc_po2 = calcDLC(po2_dy[cyc])
    #     dlc_po3 = calcDLC(po3_dy[cyc])
    #
    #     if velEgo[cyc] > velEgo_limit:
    #         if (dlc_po2 < 2.5) and (dlc_po2 > 0):
    #             near_array_po2.append(po2_dy[cyc])
    # print(near_array_po2)
    #
    # for cyc in range(len(ts_po)):
    #     # calculate dlc
    #     dlc_po2 = calcDLC(po2_dy[cyc])
    #     dlc_po3 = calcDLC(po3_dy[cyc])
    #
    #     if velEgo[cyc] > velEgo_limit:
    #         if (dlc_po3 < 2.5) and (dlc_po3 > 0):
    #             near_array_po3.append(po3_dy[cyc])
    # print(near_array_po3)
    for cyc in range(len(ts_po)-4):

        cyc_rd = 0
        for index, ts in enumerate(ts_roadedge):
            if ts > ts_po[cyc]:
                cyc_rd = index-1
                break

        # calculate dlc
        if (po2_dy[cyc] != 0) and (po2_dy[cyc+1] != 0) and (po2_dy[cyc+2] != 0) and (po2_dy[cyc+3] != 0) and (po2_dy[cyc+4] != 0):
            dtc_po2 = calcDTC(po2_dy[cyc])
        else:
            dtc_po2 = 0

        if (po3_dy[cyc] != 0) and (po3_dy[cyc+1] != 0) and (po3_dy[cyc+2] != 0) and (po3_dy[cyc+3] != 0) and (po3_dy[cyc+4] != 0):
            dtc_po3 = calcDTC(po3_dy[cyc])
        else:
            dtc_po3 = 0

        dtc_rd_le = calcDTC_rd(roadedge_le_dy[cyc_rd])
        dtc_rd_ri = calcDTC_rd(roadedge_ri_dy[cyc_rd])
        # TTC when ay = 1 m/s2
        # ay,vy,dy
        # ttc_po2 = calcTTC_left(ay_p, po2_vy[cyc], dtc_po2)
        # ttc_po3 = calcTTC_right(ay_p, po3_vy[cyc], dtc_po3)
        ttc_po2 = calcTTC(dtc_po2, day_p)
        ttc_po3 = calcTTC(dtc_po3, day_p)
        ttc_rd_le = calcTTC(dtc_rd_le, day_p)
        ttc_rd_ri = calcTTC(dtc_rd_ri, day_p)

        # distance to PO2 - distance, from 0m/s2 to Ay_target with day.
        # if rest distance < 0 -> ttc with day is the finial ttc
        # if rest distance > 0 -> finial ttc = ttc with day + ttc with Ay
        # Velocity in end of phase 1 = 1/2 * day * t^2

        # v2 = 0.5*pow(calcTTA(), 2)*day_p
        # d_po2_rest = dtc_po2 - day_p*pow(calcTTA(), 3)/6
        #
        # if d_po2_rest <= 0:
        #     ttc_po2 = ttc_po2
        # elif d_po2_rest > 0:
        #     ttc_po2 = ttc_po2 + calcT2(ay_target/2, v2, -d_po2_rest)

        # distance to PO3 - distance, from 0m/s2 to Ay_target with day.
        # if rest distance < 0 -> ttc with day is the finial ttc
        # if rest distance > 0 -> finial ttc = ttc with day + ttc with Ay
        # d_po3_rest = dtc_po3 - day_p*pow(calcTTA(), 3)/6
        #
        # if d_po3_rest <= 0:
        #     ttc_po3 = ttc_po3
        # elif d_po3_rest > 0:
        #     ttc_po3 = ttc_po3 + calcT2(ay_target/2, v2, -d_po3_rest)

        ttc_po2 = ttc_final(dtc_po2, ttc_po2)
        ttc_po3 = ttc_final(dtc_po3, ttc_po3)
        ttc_rd_le = ttc_final(dtc_rd_le, ttc_rd_le)
        ttc_rd_ri = ttc_final(dtc_rd_ri, ttc_rd_ri)

        # collisionRisk_long(ttc_po2, po2_dx[cyc], po2_vx[cyc])

        # if ttc_po2 == 0:
        #     ttc_po2_array.append(np.nan)
        # else:
        #     ttc_po2_array.append(ttc_po2)
        #
        # if ttc_po3 == 0:
        #     ttc_po3_array.append(np.nan)
        # else:
        #     ttc_po3_array.append(ttc_po3)
        # ego vehicle speed > 60 kph
        # po2 or po3 to ego vehicle edge is < 1.5 meter
        if (0.0039 * velEgo[cyc] - 0.0119) > velEgo_limit:
            near_array_total.append(po2_dy[cyc])
            # if ((dtc_po2 < dy_limit) & (dtc_po2 != 0)) | (
            #       (dtc_po3 < dy_limit) & (dtc_po3 != 0)):
            if collisionRisk_rd(po2_existprob[cyc], roadedge_le_existprob[cyc_rd], roadedge_le_dy[cyc_rd]) or collisionRisk_rd(po3_existprob[cyc], roadedge_ri_existprob[cyc_rd], roadedge_ri_dy[cyc_rd]):
                near_array.append(po2_dy[cyc])

            if (collisionRisk_long(ttc_po2, po2_dx[cyc], po2_vx[cyc])) | (collisionRisk_long(ttc_po3, po3_dx[cyc], po3_vx[cyc])):
            # if ((dtc_po2 < dy_limit) & (dtc_po2 != 0) & (collisionRisk_long(ttc_po2, po2_dx[cyc], po2_vx[cyc]))) | ((dtc_po3 < dy_limit) & (dtc_po3 != 0) & (collisionRisk_long(ttc_po3, po3_dx[cyc], po3_vx[cyc]))):
                near_array.append(po2_dy[cyc])

            if collisionRisk_long(ttc_po2, po2_dx[cyc], po2_vx[cyc]):
                ttc_po2_array.append(ttc_po2)

            if collisionRisk_rd(po2_existprob[cyc], roadedge_le_existprob[cyc_rd], roadedge_le_dy[cyc_rd]):
                ttc_po2_array.append(ttc_rd_le)

            if collisionRisk_long(ttc_po3, po3_dx[cyc], po3_vx[cyc]):
                ttc_po3_array.append(ttc_po3)

            if collisionRisk_rd(po3_existprob[cyc], roadedge_ri_existprob[cyc_rd], roadedge_ri_dy[cyc_rd]):
                ttc_po3_array.append(ttc_rd_ri)

    ttc_po2_average = np.nanmean(ttc_po2_array)
    ttc_po3_average = np.nanmean(ttc_po3_array)

    ts = ts_po[2]-ts_po[1]
    near_time_total = (len(near_array_total)*ts)
    near_time = (len(near_array)*ts)
    near_dist = 1
    # get near range time
    # po2_dy_near = po2_dy[np.where((po2_dy < 2.5) & (po2_dy > 0))]
    # po3_dy_near = po3_dy[np.where((po3_dy < 2.5) & (po3_dy > 0))]
    # near_time = (len(po2_dy_near)+len(po3_dy_near))*0.2
    # get near range distance
    # near_dist = calcDistance(po2_dy_near, velEgo)
    # --------------------------------------------------------------------------------------------------------------
    # total distance of measurement
    total_dist = calcDistance(ts_po, velEgo)
    # total time of measurement
    total_time = ts_po[-1] - ts_po[0]

    # draw plot
    # fig = plt.figure()
    # plt.title(os.path.basename(folder))
    # plt.plot(ts_po, ttc_po2_array, label='ttc po2')
    # plt.plot(ts_po, ttc_po3_array, label='ttc po3')
    # plt.gca().set_ylim(0, 1)
    # plt.legend(loc='upper right', prop={'size': 8})
    # fig.savefig(os.path.join(dirpath, os.path.basename(folder) + '_PO_dy.png'))
    # plt.close(fig)
    #
    # fig = plt.figure()
    # plt.title(os.path.basename(folder))
    # plt.plot(ts_po, po2_dx, label='PO2 dx')
    # plt.plot(ts_po, po3_dx, label='PO3 dx')
    # plt.gca().set_ylim(0, 50)
    # plt.legend(loc='upper right', prop={'size': 8})
    # fig.savefig(os.path.join(dirpath, os.path.basename(folder) + '_PO_dx.png'))
    # plt.close(fig)

    return ts_po, near_time_total, near_time, total_dist, total_time, ttc_po2_array, ttc_po3_array

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

    total_dist_all = []
    total_time_all = []
    near_time_all = []
    near_time_total_all = []
    ttc_po2_all = []
    ttc_po3_all = []
    ttc_po_all = []

    for folder in folder_list:
        # read po target distance
        try:
            mat_po = os.path.join(folder, 'po.mat')
            mat_roadedge_le = os.path.join(folder, 'roadedge_left.mat')
            mat_roadedge_ri = os.path.join(folder, 'roadedge_right.mat')
            mat_speed = os.path.join(folder, 'speed.mat')
            if os.path.exists(mat_po) and os.path.exists(mat_roadedge_le) and os.path.exists(mat_roadedge_ri) and os.path.exists(mat_speed):
                logger.info('Reading {}.'.format(folder))
            else:
                logger.warning('missing po or roadedge mat file from {}. '
                               'Make sure that it\'s a measurement folder. Skipping.'
                               .format(folder))
                continue

            m_ts_po, near_time_total, m_near_time, m_total_dist, m_total_time, m_ttc_po2_array, m_ttc_po3_array = calcDistanceKPI()

            total_dist_all.append(m_total_dist)
            total_time_all.append(m_total_time)
            near_time_all.append(m_near_time)
            near_time_total_all.append(near_time_total)
            ttc_po2_all.extend(m_ttc_po2_array)
            ttc_po3_all.extend(m_ttc_po3_array)

        except:
            pass

    ttc_po_all.extend(ttc_po2_all)
    ttc_po_all.extend(ttc_po3_all)

    ttc_po_all = np.array(ttc_po_all)
    a = len(np.where(ttc_po_all <= 1)[0])
    b = len(np.where((ttc_po_all > 1) & (ttc_po_all <= 1.5))[0])
    c = len(np.where((ttc_po_all > 1.5) & (ttc_po_all <= 2))[0])
    d = len(np.where((ttc_po_all > 2) & (ttc_po_all <= 3))[0])
    e = len(np.where(ttc_po_all > 3)[0])
    f = len(ttc_po_all)

    # print(max(ttc_po_all))
    print(a, b, c, d, e, f)

    # labels = ['0-0.5s', '0.5-1s', '1-1.5s', '1.5-2s', '2-3s', '>3s']
    # X = [222, 42, 455, 664, 454, 334]
    #
    # fig = plt.figure()
    # plt.pie(X, labels=labels, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
    # plt.title("Pie chart")
    #
    # plt.show()
    # plt.savefig("PieChart.jpg")

    fig = plt.figure()
    plt.title(os.path.basename(folder))

    labels = ['<1s', '1-1.5s', '1.5-2s', '2-3s', '>3s']
    X = [a, b, c, d, e]
    explode = (0.1, 0.1, 0.02, 0.02, 0.02)  # 将某一块分割出来，值越大分割出的间隙越大

    fig = plt.figure()
    plt.pie(X, labels=labels, explode=explode, autopct='%1.2f%%', pctdistance=0.8, labeldistance=1.2, startangle=30)
    plt.title("PO ttc chart")

    fig.savefig(os.path.join(dirpath, 'PO_ttc.png'))
    plt.close(fig)

    print("---------------------------------------------------")
    print("total distance:", "%.2f" % (np.sum(total_dist_all)/1000), "km")
    print("total time:", "%.2f" % (np.sum(total_time_all)/3600), "hours")
    print("high speed time (V>60kph):", "%.4f" % (np.sum(near_time_total_all)/3600), "hours")
    print("collision risky time:", "%.4f" % (np.sum(near_time_all)/3600), "hours")
    print("---------------------------------------------------")
    print("high speed time in total time:", "%.2f" % (np.sum(near_time_total_all)/np.sum(total_time_all)*100), "%")
    print("collision risky time percent in total time:", "%.2f" % (np.sum(near_time_all) / np.sum(total_time_all) * 100), "%")
    print("collision risky time percent in high speed time:", "%.2f" % (np.sum(near_time_all)/np.sum(near_time_total_all)*100), "%")
    print("---------------------------------------------------")
    print("average TTC po2 (day=", day_p, "m/s3):", "%.2f" % (np.nanmean(ttc_po2_all)), "second")
    print("average TTC po3 (day=", day_p, "m/s3):", "%.2f" % (np.nanmean(ttc_po3_all)), "second")
    print("TTC < 1.5", "%.2f" % ((a+b)/f*100), "%")
    print("TTC = 1.5-2", "%.2f" % (c/f*100), "%")
    print("TTC = 2-3", "%.2f" % (d/f*100), "%")
    print("TTC > 3", "%.2f" % (e/f*100), "%")
    # fig = plt.figure()
    # plt.title(os.path.basename(folder))
    # plt.plot(range(len(ttc_po2_all)), ttc_po2_all, label='ttc po2')
    # plt.gca().set_ylim(0, 1)
    # plt.legend(loc='upper right', prop={'size': 8})
    # fig.savefig(os.path.join(dirpath, os.path.basename(folder) + '_PO2_ttc.png'))
    # plt.close(fig)





