import glob
import logging
import os
import sys
from logging import Logger, RootLogger
from typing import Any, Union

import seaborn as sns
# from numpy.random import randn
# import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

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
# import matplotlib.pyplot as plt
config = ConfigParser()

#-------------------------------------
# EXCEL
from datetime import datetime
import xlsxwriter
#-------------------------------------

logger: Union[Union[Logger, RootLogger], Any] = logging.getLogger('GPSHeading')
logging.basicConfig(level=logging.INFO)

config.read('constantsMAT.config', encoding='UTF-8')

# ego vehicle parameters
vehicle_halfWide = 1    # unit - meter
vehicle_halfWide_po = 1  # unit - meter
vehicle_length = 4  # unit - meter
vehicle_length_po = 4  # unit - meter
vehicle_trunk = 1 # unit - meter 车辆后备箱纵向长度

# Evaluation threshold
ego_speedLimit = 60  # unit - km/h
dy_limit = 3.5  # unit - meter
ay_target = 0  # unit - m/s2 加速度，set to 0
day_p = 999  # unit - m/s3 加速度变化率,set to 999
dx_farLimit = 50  # unit - meter
dx_nearLimit = 0  # unit - meter, set to 0

#自车偏移参数
vy_l = 0.4 # unit - m/s 自车侧偏的横向速度
m_l = 0.7 # unit - m 自车侧偏m_l时，后车开始减速
t0 = m_l/vy_l # unit - s 自车侧偏m_l用时，本身无用，这个时间段两车运动状态不变，可以直接用dy-m_l代入

#自车偏移时，侧后方行车参数
apo_b = -6 # unit m/s2, 侧后车匀减速的加速度
apo_f = 0 # unit m/s2, 侧后车匀加速的加速度
"""
须知：定义index来表征目标物是否切换，而限制这个值有个数字是32（31）,注意适配
"""
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
    time_diff = np.append(np.array(0), np.diff(time))
    distance = np.cumsum(np.multiply(time_diff, vel))
    total_dist = distance[-1]
    total_dist = abs(total_dist)

    return total_dist


def calcDTC(dy_po): # when - m_l, the moment POX start deceleration
    if dy_po > 0:
        dtc = dy_po - vehicle_halfWide - vehicle_halfWide_po # t0_l 时刻的距离
    else:
        dtc = -dy_po - vehicle_halfWide - vehicle_halfWide_po # t0_l 时刻的距离

    return dtc
    # dtc是两车之间的空隙的横向长度，这里已经处理为正值


def collisionV(ttc, dx_po, vx_po):
    if (dx_po < -1):
        dx_po = abs(dx_po + vehicle_trunk)
        t1 = ttc - t0
        distance_ttc = vx_po * ttc
        distance_t0 = vx_po*t0
        distance_t1 = vx_po * t1 + 0.5 * apo_b * pow(t1, 2)

        vx_end = vx_po + apo_b*t1
        distance_t2 = 0.5*vx_end*vx_end/abs(apo_b)

        #distance_po_acc = dx_po + vx_po * ttc + 0.5 * apo_f * pow(ttc, 2)
        # if collision in t0
        if(ttc <= t0 and ttc > 0):
            if(distance_ttc < dx_po): # rear collision: after ttc, distance to po > 0
                if(distance_t0 > dx_po):
                    return vx_po
        # if collision in t0-t1
        elif(ttc > t0):
            if(distance_t0 + distance_t1 < dx_po):# rear collision, after ttc, distance to po > 0
                if(distance_t0 + distance_t1 + distance_t2 > dx_po):
                    vx = pow((2*apo_b*(dx_po-distance_t0) + vx_po*vx_po), 0.5) # Vx^2 - Vx_po^2 = 2*apo_b*(dx_po-distance_t0)
                    return vx

def collisionRisk_long(ttc, dx_po, vx_po, apo_f, apo_b):
    # ttc_l 为两车横向距离减为0经过的时间
    # dx_po 为t0_l时刻两车纵向距离
    # vx_po 为相对速度，以自车为参考系
    # apo_b 为后车减速的加速度，前文定义为负值
    if (dx_po < -1):
        dx_po = abs(dx_po + vehicle_trunk)
        t1 = ttc - t0
        distance_ttc = vx_po * ttc
        distance_t0 = vx_po*t0
        distance_t1 = vx_po * t1 + 0.5 * apo_b * pow(t1, 2)

        vx_end = vx_po + apo_b*t1
        distance_t2 = 0.5*vx_end*vx_end/abs(apo_b)

        # 原纵向距离减去相对位移，后车比前车快，则得负
        # distance_op 为ttc_l时刻纵向的剩余距离，负值表示后车在后面，正值表示后车前保险杠已过自车后轴
        # dx_po一般是负值，后车比前车快时vx_po为正值，apo_l定义为负值
        #distance_po_acc = dx_po + vx_po * ttc + 0.5 * apo_f * pow(ttc, 2)
        # if collision in t0
        if(ttc <= t0 and ttc > 0):
            if(distance_ttc > dx_po and distance_ttc < dx_po + 8): # side collision
                # return True
                return False # rear collision only
            elif(distance_ttc < dx_po): # rear collision: after ttc, distance to po > 0
                if(distance_t0 > dx_po):
                    return True
                else:
                    return False
            else:
                return False
        # if collision in t0-t1
        elif(ttc > t0):
            if(distance_t0 + distance_t1 >= dx_po and distance_t0 + distance_t1 < dx_po + 8): # side collision
                # return True
                return False  # rear collision only
            elif(distance_t0 + distance_t1 < dx_po):# rear collision, after ttc, distance to po > 0
                if(distance_t0 + distance_t1 + distance_t2 > dx_po):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False
    # if(dx_po + distance_t0 + distance_t1 >= 0 and dx_po + distance_t0 + distance_t1 < 8):
    #     return True
    # elif(vx_end > 0):
    #     t2 = abs(vx_end/apo_b)
    #     distance_to_po = dx_po + distance_t0 + distance_t1
    #     distance_t2 = 0.5*apo_b*t2*t2
    #     if (distance_to_po-distance_t2 > 0):
    #         return True
    #     else:
    #         return False
    # else:
    #     return False
        
        

    
    # if (ttc_l == t0):
    #     # t0 时刻，两车已横向交叠
    #     t2 = vx_po/(-apo_b)
    #     # t2 表征两车由ttc_l到达等速的时间，相对速度归0，注意验证合理性，vx_po为正才有意义
    #     if (vx_po <= 0):
    #         return False
    #     elif (distance_po_dec + vehicle_trunk < (0.5 * apo_b * pow(t2, 2))):
    #         # 等速时，距离仍 > 0
    #         return False
    #     else:
    #         deltaV_collision = vx_po + apo_b*ttc_l
    #         return True # 等速时，距离 < 0       
    # elif (distance_po_acc > (2 * vehicle_length - vehicle_trunk)):
    #     # 判断后车位置，完全超过自车，不碰撞。考虑后车匀速或加速。
    #     # 雷达采样以后轴中心位置为原点。超过，考虑自车后轴到目标车前保险杠。
    #     return False
    # elif (distance_po_dec < (-vehicle_trunk)):
    #     # 判断后车位置，未及，但可能后车保持减速碰撞。未及，考虑后备箱纵向长度。
    #     t2 = vx_po/apo_b - ttc_l
    #     # ttc_l + t2 表征两车等速的时间，相对速度归0
    #     if (vx_po <= 0):
    #         return False #后车速度已比自车慢
    #     elif (distance_po_dec < (0.5 * apo_b * pow(t2, 2))):
    #         # calcT2(a,b,c)
    #         return False # 等速时，距离仍>=0
    #     else: # 两个负距离中，distance较大，视为减速状态追尾
    #         deltaV_collision = vx_po + apo_b*ttc_l
    #         return True
    # else:
    #     return False
    
# 原函数 calcTTC calcTTC_l
def calcTTC_l_left(dtc, vy, vy_l): 
    # 计算自车匀速走过两车之间间隙的横向距离的时间
    # dtc 在其子函数修正为正值，vy 保留正负值含义，vy_l 定义为正值
    # 输出ttc_l为正值或0
    if dtc > 0:
        ttc_l = abs(dtc)/(vy_l)
    else:
        ttc_l = 0
    return ttc_l

def calcTTC_l_right(dtc, vy, vy_l): 
    # 计算自车匀速走过两车之间间隙的横向距离的时间
    # dtc 在其子函数修正为正值，vy 保留正负值含义，vy_l 定义为正值
    # 输出ttc_l为正值或0
    if dtc != 0:
        ttc_l = abs(dtc)/(vy_l)
    else: 
        ttc_l = 0
    return ttc_l

"""   
def calcTTA():
    t = ay_target/day_p

    return t
"""
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

"""
def calcTTC_left(ay,vy,dy,a_l): # 加入a_l作为函数输入
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

def calcTTC_right(ay,vy,dy,a_l): # 加入a_l作为函数输入
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
"""
def calcDistanceKPI():
    near_array = []
    near_array_total = []
    # near_array_po2 = []
    # near_array_po3 = []
    ttc_po2_array = []
    dtc_po2_array = []
    vx_po2_array = []
    ttc_po3_array = []
    dtc_po3_array = []
    vx_po3_array = []
    dx_po2_loc = []
    vx_po2_loc = []
    dx_po3_loc = []
    vx_po3_loc = []
    v_collision_array =[]


    mat_RRL_00 = sio.loadmat(mat_file_path_RRL_00, struct_as_record=False, squeeze_me=True)
    mat_RRL_01 = sio.loadmat(mat_file_path_RRL_01, struct_as_record=False, squeeze_me=True)
    mat_RRR_00 = sio.loadmat(mat_file_path_RRR_00, struct_as_record=False, squeeze_me=True)
    mat_RRR_01 = sio.loadmat(mat_file_path_RRR_01, struct_as_record=False, squeeze_me=True)
    mat_VehSpd = sio.loadmat(mat_file_path_VehSpd, struct_as_record=False, squeeze_me=True)
    
    
    matDictFus_RRL_00 = buildDict(mat_RRL_00)
    matDictFus_RRL_01 = buildDict(mat_RRL_01)
    matDictFus_RRR_00 = buildDict(mat_RRR_00)
    matDictFus_RRR_01 = buildDict(mat_RRR_01)
    matDictFus_VehSpd = buildDict(mat_VehSpd)
    
    ts_po_L = getMatSignal("timestamp", matDictFus_RRL_00)
    ts_po_R = getMatSignal("timestamp", matDictFus_RRR_00)
    
    VehSpd = getMatSignal("vehicle_speed", matDictFus_VehSpd) # Unit km/h
    
    
    if len(ts_po_L) > len(ts_po_R):
        ts_po = ts_po_R
    else:
        ts_po = ts_po_L

    # get information from mat file
    # velEgo = getMatSignal("ego;vxvRef", matDictFus)
    po2_index = getMatSignal("RRL;Obj00_ID", matDictFus_RRL_00)
    po2_dx = getMatSignal("RRL;Obj00_Dx", matDictFus_RRL_00)
    po2_dy = getMatSignal("RRL;Obj00_Dy", matDictFus_RRL_00)
    po2_vx = getMatSignal("RRL;Obj00_Vx", matDictFus_RRL_00)
    po2_vy = getMatSignal("RRL;Obj00_Vy", matDictFus_RRL_00)
    
    po3_index = getMatSignal("RRR;Obj00_ID", matDictFus_RRR_00)
    po3_dx = getMatSignal("RRR;Obj00_Dx", matDictFus_RRR_00)
    po3_dy = getMatSignal("RRR;Obj00_Dy", matDictFus_RRR_00)
    po3_vy = getMatSignal("RRR;Obj00_Vx", matDictFus_RRR_00)  # 0-32 row
    po3_vx = getMatSignal("RRR;Obj00_Vy", matDictFus_RRR_00)  # 0-32 row

    # --------------------------------------------------------------------------------------------------------------
    # get po2 and po3 vy according to index
    """
    po2_vy = getPOvy(ts_po, po2_index, po_vy)
    po3_vy = getPOvy(ts_po, po3_index, po_vy)

    po2_vx = getPOvx(ts_po, po2_index, po_vx)
    po3_vx = getPOvx(ts_po, po3_index, po_vx)
    """
    # --------------------------------------------------------------------------------------------------------------
    # get v > 60 kph (16.67 m/s) and dy < 2.5m
    # velEgo_limit = ego_speedLimit /3.6  # (config["ego_vehicle"].getint("ego_speedLimit"))

    # total time of measurement
    total_time = ts_po[-1] - ts_po[0] 
    # total distance of measurement
    total_dist = sum(VehSpd)/len(VehSpd) * total_time/3600 # 一段数据内的公里数


    for cyc in range(len(ts_po)-4):    
        if (po2_vx[cyc] < -20 or po2_vx[cyc] > 20):
            po2_vx[cyc] = 0
        if (po3_vx[cyc] < -20 or po3_vx[cyc] > 20):
            po3_vx[cyc] = 0
        # calculate dtc
        # filter: dtc > 2* half vehicle wide
        # po2_dy[cyc] = 3.6 # fix dy
        # po3_dy[cyc] = 3.6 # fix dy

        if (abs(po2_dy[cyc]) > vehicle_halfWide+vehicle_halfWide_po) and \
                (po2_dy[cyc] != 0) and (po2_dy[cyc+1] != 0) and (po2_dy[cyc+2] != 0) and (po2_dy[cyc+3] != 0) and (po2_dy[cyc+4] != 0):
            dtc_po2 = calcDTC(po2_dy[cyc])
            # dtc 修正为t0_l时刻，即 - m_l
            dx_po2_loc.append(po2_dx[cyc])
            vx_po2_loc.append(po2_vx[cyc])
            # 记录dx，vx用于统计作图
        else:
            dtc_po2 = -1
            

        if (abs(po3_dy[cyc]) > vehicle_halfWide+vehicle_halfWide_po) and \
                (po3_dy[cyc] != 0) and (po3_dy[cyc+1] != 0) and (po3_dy[cyc+2] != 0) and (po3_dy[cyc+3] != 0) and (po3_dy[cyc+4] != 0):
            dtc_po3 = calcDTC(po3_dy[cyc])
            # dtc 修正为t0_l时刻，即 - m_l
            dx_po3_loc.append(po3_dx[cyc])
            vx_po3_loc.append(po3_vx[cyc])
            # 记录dx，vx用于统计作图
        else:
            dtc_po3 = -1
            
            
        # 由t0_l至两车横向距离为0的时间，ttc_l
        ttc_po2 = calcTTC_l_left(dtc_po2, po2_vy[cyc], vy_l) #注意适配
        ttc_po3 = calcTTC_l_right(dtc_po3, po3_vy[cyc], vy_l) #注意适配
        
        """
        #v2 = 0.5*pow(calcTTA(), 2)*day_p
        #d_po2_rest = dtc_po2 - day_p*pow(calcTTA(), 3)/6

        #if d_po2_rest <= 0:
        #    ttc_po2 = ttc_po2
        #elif d_po2_rest > 0:
        #    ttc_po2 = ttc_po2 + calcT2(ay_target/2, v2, -d_po2_rest)
        #d_po3_rest = dtc_po3 - day_p*pow(calcTTA(), 3)/6

        #if d_po3_rest <= 0:
        #    ttc_po3 = ttc_po3
        #elif d_po3_rest > 0:
        #   ttc_po3 = ttc_po3 + calcT2(ay_target/2, v2, -d_po3_rest)
        """
        #if velEgo[cyc] > velEgo_limit:
        near_array_total.append(po2_dy[cyc]) # 只是用到长度
            # if ((dtc_po2 < dy_limit) & (dtc_po2 != 0)) | (
            #       (dtc_po3 < dy_limit) & (dtc_po3 != 0)):

        if (collisionRisk_long(ttc_po2, po2_dx[cyc], po2_vx[cyc], apo_f, apo_b)) | (collisionRisk_long(ttc_po3, po3_dx[cyc], po3_vx[cyc], apo_f, apo_b)):
            near_array.append(po2_dy[cyc]) # # 只是用到长度

        if collisionRisk_long(ttc_po2, po2_dx[cyc], po2_vx[cyc], apo_f, apo_b):
            ttc_po2_array.append(ttc_po2)
            dtc_po2_array.append(po2_dx[cyc])
            vx_po2_array.append(po2_vx[cyc])
            v_collision_array.append(collisionV(ttc_po2, po2_dx[cyc], po2_vx[cyc]))

        if collisionRisk_long(ttc_po3, po3_dx[cyc], po3_vx[cyc], apo_f, apo_b):
            ttc_po3_array.append(ttc_po3)
            dtc_po3_array.append(po3_dx[cyc])
            vx_po3_array.append(po3_vx[cyc])
            v_collision_array.append(collisionV(ttc_po3, po3_dx[cyc], po3_vx[cyc]))

    ttc_po2_average = np.nanmean(ttc_po2_array)
    ttc_po3_average = np.nanmean(ttc_po3_array)

    ts = ts_po[2]-ts_po[1]
    near_time_total = (len(near_array_total)*ts)
    near_time = (len(near_array)*ts)
    # near_dist = 1
    # get near range time
    # po2_dy_near = po2_dy[np.where((po2_dy < 2.5) & (po2_dy > 0))]
    # po3_dy_near = po3_dy[np.where((po3_dy < 2.5) & (po3_dy > 0))]
    # near_time = (len(po2_dy_near)+len(po3_dy_near))*0.2
    # get near range distance
    # near_dist = calcDistance(po2_dy_near, velEgo)
    # --------------------------------------------------------------------------------------------------------------
    
    return ts_po, near_time_total, near_time, total_time, total_dist, ttc_po2_array, ttc_po3_array, dx_po2_loc, vx_po2_loc, dx_po3_loc, vx_po3_loc, dtc_po2_array, dtc_po3_array, vx_po2_array, vx_po3_array, v_collision_array

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
    dx_po2_all = []
    vx_po2_all = []
    dx_po3_all = []
    vx_po3_all = []
    vc_all = []

    dx_all = []
    vx_all = []
    
    dx_po2_c_all = []
    vx_po2_c_all = []
    dx_po3_c_all = []
    vx_po3_c_all = []
    
    
    for folder in folder_list:
        # read po target distance
        mat_file_path_RRL_00 = os.path.join(folder, 'RRL_00.mat')
        mat_file_path_RRL_01 = os.path.join(folder, 'RRL_01.mat')
        mat_file_path_RRR_00 = os.path.join(folder, 'RRR_00.mat')
        mat_file_path_RRR_01 = os.path.join(folder, 'RRR_01.mat')
        mat_file_path_VehSpd = os.path.join(folder, 'VehSpd.mat')
        
        if os.path.exists(mat_file_path_RRL_00):
            logger.info('Reading {}.'.format(folder))
        else:
            logger.warning('objects_fused.mat AND/OR GPS_GTLines.csv AND/OR GPS_Pos_Heading.csv is missing from {}. '
                           'Make sure that it\'s a measurement folder. Skipping.'
                           .format(folder))
            continue

        m_ts_po, near_time_total, m_near_time, m_total_time, m_total_dist, m_ttc_po2_array, m_ttc_po3_array, dx_po2_loc, vx_po2_loc, dx_po3_loc, vx_po3_loc, dx_po2_c, dx_po3_c, vx_po2_c, vx_po3_c, v_c = calcDistanceKPI()

        total_dist_all.append(m_total_dist)
        total_time_all.append(m_total_time)
        near_time_all.append(m_near_time)
        near_time_total_all.append(near_time_total)
        ttc_po2_all.extend(m_ttc_po2_array)
        ttc_po3_all.extend(m_ttc_po3_array)
        
        # 统计目标物的dx，vx，用于作图
        dx_po2_all.extend(dx_po2_loc)
        vx_po2_all.extend(vx_po2_loc)
        dx_po3_all.extend(dx_po3_loc)
        vx_po3_all.extend(vx_po3_loc)
        # collision time all
        dx_po2_c_all.extend(dx_po2_c)
        vx_po2_c_all.extend(vx_po2_c)
        dx_po3_c_all.extend(dx_po3_c)
        vx_po3_c_all.extend(vx_po3_c)
        vc_all.extend(v_c)


    ttc_po_all.extend(ttc_po2_all)
    ttc_po_all.extend(ttc_po3_all)
    ttc_po_all = np.array(ttc_po_all)
    vc_alldata = np.array(vc_all)
    dx_all.extend(dx_po2_all)
    dx_all.extend(dx_po3_all)
    vx_all.extend(vx_po2_all)
    vx_all.extend(vx_po3_all)

    #print(vc_alldata)
    vc_alldata = [x for x in vc_alldata if str(x) != 'None']
    a = len([x for x in vc_alldata if x <= 10])
    b = len([x for x in vc_alldata if (x > 10 and x <= 50)])
    c = len([x for x in vc_alldata if x < 0])
    f = len(vc_alldata)

    # # print(max(ttc_po_all))
    # print(a, b, c, d, e, f)

    # labels = ['0-0.5s', '0.5-1s', '1-1.5s', '1.5-2s', '2-3s', '>3s']
    # X = [222, 42, 455, 664, 454, 334]
    #
    # fig = plt.figure()
    # plt.pie(X, labels=labels, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
    # plt.title("Pie chart")
    #
    # plt.show()
    # plt.savefig("PieChart.jpg")

    # fig = plt.figure()
    # plt.title(os.path.basename(folder))

    # labels = ['<1s', '1-1.5s', '1.5-2s', '2-3s', '>3s']
    # X = [a, b, c, d, e]
    # explode = (0.1, 0.1, 0.02, 0.02, 0.02)  # 将某一块分割出来，值越大分割出的间隙越大

    # fig = plt.figure()
    # plt.pie(X, labels=labels, explode=explode, autopct='%1.2f%%', pctdistance=0.8, labeldistance=1.2, startangle=30)
    # plt.title("PO ttc chart")

    # fig.savefig(os.path.join(dirpath, 'PO_ttc.png'))
    # plt.close(fig)

    print("---------------------------------------------------")
    print("total distance:", "%.2f" % (np.sum(total_dist_all)), "km")
    print("total time:", "%.2f" % (np.sum(total_time_all)/3600), "hours")
    # print("high speed time (V>60kph):", "%.4f" % (np.sum(near_time_total_all)/3600), "hours")
    # print("collision risky time:", "%.4f" % (np.sum(near_time_all)/3600), "hours")
    print("---------------------------------------------------")
    # print("high speed time in total time:", "%.2f" % (np.sum(near_time_total_all)/np.sum(total_time_all)*100), "%")
    print("collision risky time percent in total time:", "%.2f" % (np.sum(near_time_all) / np.sum(total_time_all) * 100), "%")
    # print("collision risky time percent in high speed time:", "%.2f" % (np.sum(near_time_all)/np.sum(near_time_total_all)*100), "%")
    print("---------------------------------------------------")
    print("average TTC RL (day=", day_p, "m/s3):", "%.2f" % (np.nanmean(ttc_po2_all)), "second")
    print("average TTC RR (day=", day_p, "m/s3):", "%.2f" % (np.nanmean(ttc_po3_all)), "second")
    print("deltaV when rear collision < 10kph", "%.2f" % (a/f*100), "%")
    print("deltaV when rear collision > 10kph and < 50kph", "%.2f" % (b/f*100), "%")
    print("deltaV when rear collision < 0", "%.2f" % (c/f*100), "%")
    # print("TTC > 3", "%.2f" % (e/f*100), "%")
    # fig = plt.figure()
    # plt.title(os.path.basename(folder))
    # plt.plot(range(len(ttc_po2_all)), ttc_po2_all, label='ttc po2')
    # plt.gca().set_ylim(0, 1)
    # plt.legend(loc='upper right', prop={'size': 8})
    # fig.savefig(os.path.join(dirpath, os.path.basename(folder) + '_PO2_ttc.png'))
    # plt.close(fig)
    # ----------------------------------------------------------------------------------------------------    
    # 分布图
    # style set 这里只是一些简单的style设置
    sns.set_palette('deep', desat=.6)
    current_palette = sns.color_palette("hls",12)
    sns.set_style("darkgrid")
    sns.set_context(rc={'figure.figsize': (8, 15)})
    x = dx_po2_c_all
    y = vx_po2_c_all

    data = {'Dx': x, 'Vx': y}
    df = pd.DataFrame(data)
    # sns_plot = sns.jointplot(data=df, x="Dx", y="Vx", )
    sns_plot = sns.jointplot(data=df, x="Dx", y="Vx", color='#FF0000', kind='hex')
    #sns_plot.displot(data=df, stat = "density")
    sns_plot.savefig(os.path.join(dirpath, 'Dx-Vx_distribution.png'))
    #plt.show()

    # all data
    sns.set_palette('deep', desat=.6)
    current_palette = sns.color_palette("hls",12)
    sns.set_style("darkgrid")
    sns.set_context(rc={'figure.figsize': (8, 15)})
    x = dx_po2_all
    y = vx_po2_all
    x_c = dx_po2_c_all
    y_c = vx_po2_c_all

    data = {'Dx': x, 'Vx': y}
    df = pd.DataFrame(data)
    # sns_plot = sns.jointplot(data=df, x="Dx", y="Vx", )
    sns_plot = sns.jointplot(data=df, x="Dx", y="Vx", kind='hex')
    sns_plot.savefig(os.path.join(dirpath, 'Dx-Vx_distribution_all.png'))

    sns_plot = sns.displot(data=x,kind = "hist",hue_norm = [0.0, 1.0], stat = "density")
    sns_plot.savefig(os.path.join(dirpath, 'Dx_hist.png'))

    sns_plot = sns.displot(data=y,kind = "hist",hue_norm = [0.0, 1.0], stat = "density")
    sns_plot.savefig(os.path.join(dirpath, 'Vx_hist.png'))

    sns_plot = sns.displot(data=x_c, color='#FF0000', kind = "hist",hue_norm = [0.0, 1.0], stat = "density")
    sns_plot.savefig(os.path.join(dirpath, 'Dx_c_hist.png'))

    sns_plot = sns.displot(data=y_c, color='#FF0000', kind = "hist",hue_norm = [0.0, 1.0], stat = "density")
    sns_plot.savefig(os.path.join(dirpath, 'Vx_c_hist.png'))


    # sns.set_palette('deep', desat=.6)
    # current_palette = sns.color_palette("hls",12)
    # sns.set_style("ticks")
    # sns.set_context(rc={'figure.figsize': (8, 15) } )
    # # np.random.seed(1425)
    # # figsize是常用的参数.
    # # x = stats.gamma(2).rvs(5000)
    # # print(dx_po2_all)
    # x_all = dx_po2_all
    # y_all = vx_po2_all
    # x_c = dx_po2_c_all
    # y_c = vx_po2_c_all
    # array1 = list(zip(x_all,y_all))
    # array2 = list(zip(x_c,y_c))
    # # print(array1)
    # dxvx = array1
    # with sns.axes_style("dark"):
    #     sns_plot = sns.jointplot(data = dxvx, x="Dx", y="Vx", kind='hex')
    #     sns_plot.savefig(os.path.join(dirpath, 'Dx-Vx_distribution.png'))
    # plt.show()
    
    # # collision distribution 
    # sns.set_palette('deep', desat=.6)
    # current_palette = sns.color_palette("hls",12)
    # sns.set_style("ticks")
    # sns.set_context(rc={'figure.figsize': (8, 15) } )
    # # np.random.seed(1425)
    # # figsize是常用的参数.
    # # x = stats.gamma(2).rvs(5000)
    # # print(dx_po2_all)

    # # xlim = [-120,10]
    # # ylim = [-25,25]
    # x_max = max(dx_po2_all)
    # y_max = max(vx_po2_all)
    # x_min = min(dx_po2_all)
    # y_min = min(vx_po2_all)
    # dx_po2_c_all.append(x_max)
    # vx_po2_c_all.append(y_max)
    # dx_po2_c_all.append(x_min)
    # vx_po2_c_all.append(y_min)
    #
    # x = dx_po2_c_all
    # y = vx_po2_c_all
    
    # with sns.axes_style("dark"):
    #     sns_plot = sns.jointplot(x, y, color='#FF0000', kind='hex')
    #     sns_plot.savefig(os.path.join(dirpath, 'Dx-Vx_distribution_collision.png'))
    # plt.show()
    # ----------------------------------------------------------------------------------------------------
    
    # workbook = xlsxwriter.Workbook('Dx-Dy.xlsx')
    # worksheet = workbook.add_worksheet()
    #
    # # Add a bold format to use to highlight cells.
    # bold = workbook.add_format({'bold': 1})
    # # Adjust the column width.
    # worksheet.set_column(1, 1, 15)
    # # Write some data headers.
    # worksheet.write('A1', 'Dx', bold)
    # worksheet.write('B1', 'Vx', bold)
    # # Start from the first cell below the headers.
    # row = 1
    # col = 0
    # for n in range(len(x)):
    #     worksheet.write_number (row, col, x[n] )
    #     worksheet.write_number (row, col + 1, y[n])
    #     row += 1
    # workbook.close()
    # #------------------------------------------------------------------------------------------------------
    # dx_rc = []
    # for cyc in range(len(x)):
    #     dx_rc[cyc] = vx_po2_all[cyc]*t0 + vx_po2_all[cyc]*ttc_po2_all[cyc]+0.5*apo_b*(pow(ttc_po2_all[cyc],2))
    #     if dx_rc[cyc] < 0:
    #         dx_rc[cyc] = 0
    # print("average DTC po2 (apo_b", apo_b, "m/s2):", "%.2f" % (np.nanmean(dx_rc)), "m")