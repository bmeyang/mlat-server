import numpy as np
import math
import itertools
import time

from mlat import geodesy, constants
from mlat.server import config, solver

'''
伪逆法求解时差定位的非线性方程组
'''


def tdoa_pinv_solve(measurements, altitude=10000, altitude_error=None):
    '''

    :param measurements:
    :param altitude:
    :param altitude_error:
    :return: 多组解组成的list。需要进一步处理：选取相关解、数据融合
             格式：[[[x11, y11, z11], [x12, y12, z12]], [[x21, y21, z21], [x22, y22, z22]]]
    '''
    if len(measurements) < 4:
        raise ValueError('Not enough measurements available for pseudo inverse algorithm')
    base_timestamp = measurements[0][1]
    pseudorange_data = [[receiver,  # 实际传入的参数可能是类对象，需要receiver.positon
                         # pseudorange_data = [[receiver.position,  # 实际传入的参数可能是类对象，需要receiver.positon
                         # (timestamp - base_timestamp) * 1e6,  # s -> us
                         (timestamp - base_timestamp),  # us
                         math.sqrt(variance) * constants.Cair]
                        for receiver, timestamp, variance in measurements]
    group_data = list(itertools.combinations(pseudorange_data, 4))  # 求组合
    group_num = len(group_data)  # 组合个数
    i = 0
    result = []
    while i < group_num:
        dd = list(group_data[i])
        dd.sort(key=lambda x: x[1])
        rr = tdoa_pinv(dd)
        i += 1
        result.append(rr)
    # print("tdoa_pinv_solve求解=", result)
    return result


def related_solutions(position, radius=config.MAX_RADIUS):
    '''
    选取相关解
    适应于基站个数>=5，采用了基站分组求解算法的情况
    :param position: tdoa_pinv_solve的返回值，所有分组解的list
    :radius:解模糊时采用的半径
    :return: 相关解
    '''

    num = len(position)
    if num <= 1:  # 若只有一组解，无法选取相关解
        return [position[0][0], position[0][1]]

    result = []
    center = []

    i = 0
    while i < num:  # 找可用的中心点
        if (position[i][0] == [0, 0, 0]) ^ (position[i][1] == [0, 0, 0]):
            break
        i += 1

    if i >= num:  # 全是无解或者全都是模糊解
        if position[0][0] == [0, 0, 0]:  # 全是无解
            return [[0, 0, 0]]
        else:  # 全是模糊解
            pos = [position[0][0], position[0][1], position[1][0], position[1][1]]
            distance = [geodesy.ecef_distance(pos[0], pos[2]), geodesy.ecef_distance(pos[1], pos[2]),
                        geodesy.ecef_distance(pos[0], pos[3]), geodesy.ecef_distance(pos[1], pos[3])]
            min_dis = min(distance)
            idx = distance.index(min_dis)
            if idx == 0:
                center = pos[0]
            elif idx == 1:
                center = pos[2]
            elif idx == 2:
                center = pos[3]
            elif idx == 3:
                center = pos[1]
    else:  # 存在一组解，其索引为i，其中一个解是无解[0,0,0]，另一个是真值
        center = [position[i][0][j] + position[i][1][j] for j in range(3)]  # 中心点坐标

    k = 0
    while k < num:
        if position[k][0] != [0, 0, 0]:
            if geodesy.ecef_distance(position[k][0], center) <= radius:
                result.append(position[k][0])
        if position[k][1] != [0, 0, 0]:
            if geodesy.ecef_distance(position[k][1], center) <= radius:
                result.append(position[k][1])
        k += 1

    return result


def fusion_solution(related_position):
    '''

    :param related_position: 相关解 格式：[[x1,y1,z1], [x2,y2,z2]]
    :return: 相关解经过数据融合之后得到的唯一一组解
    '''

    num = len(related_position)  # 相关解的个数
    if num <= 1:
        return [related_position[0][0], related_position[0][1], related_position[0][2]]
    else:
        for jj in related_position:
            if jj == [0, 0, 0]:
                related_position.remove(jj)
        noneZeroNum = len(related_position)
        if noneZeroNum <= 0:  # 全是[0,0,0]
            return [0, 0, 0]
        elif noneZeroNum == 1:
            return related_position[0]
        else:
            xx = [pos[0] for pos in related_position]
            yy = [pos[1] for pos in related_position]
            zz = [pos[2] for pos in related_position]

            # result = [np.mean(xx), np.mean(yy), np.mean(zz)] #暂时采用简化方法（平均值），需要进一步完善，例如建立支持度矩阵求加权系数，再加权平均

            # 建立x的支持度矩阵，计算x的融合结果
            x_dij = []
            for i in xx:
                for j in xx:
                    x_dij.append(abs(i - j))
            max_xx = max(x_dij)
            x_sij = []
            for dd in x_dij:
                x_sij.append(1 - dd / max_xx)
            x_sij = np.reshape(x_sij, (num, num))  # 支持度矩阵
            x_eval, x_evec = np.linalg.eig(x_sij)  # 计算特征值、特征向量
            x_max_eval_idx = np.argsort(x_eval)[-1]  # 最大模特征值对应的索引
            x_V = x_evec[:, x_max_eval_idx:x_max_eval_idx + 1]  # 最大模特征值对应的特征向量
            sum_xV = sum(x_V)
            x_W = np.array([x_V_v / sum_xV for x_V_v in x_V])
            x_pos = np.dot(xx, x_W)[0]  # 融合后的x坐标点

            # 建立y的支持度矩阵，计算y的融合结果
            y_dij = []
            for i in yy:
                for j in yy:
                    y_dij.append(abs(i - j))
            max_yy = max(y_dij)
            y_sij = []
            for dd in y_dij:
                y_sij.append(1 - dd / max_yy)
            y_sij = np.reshape(y_sij, (num, num))  # 支持度矩阵
            y_eval, y_evec = np.linalg.eig(y_sij)  # 计算特征值、特征向量
            y_max_eval_idx = np.argsort(y_eval)[-1]  # 最大模特征值对应的索引
            y_V = y_evec[:, y_max_eval_idx:y_max_eval_idx + 1]  # 最大模特征值对应的特征向量
            sum_yV = sum(y_V)
            y_W = np.array([y_V_v / sum_yV for y_V_v in y_V])
            y_pos = np.dot(yy, y_W)[0]  # 融合后的y坐标点

            # 建立z的支持度矩阵，计算z的融合结果
            z_dij = []
            for i in zz:
                for j in zz:
                    z_dij.append(abs(i - j))
            max_zz = max(z_dij)
            z_sij = []
            for dd in z_dij:
                z_sij.append(1 - dd / max_zz)
            z_sij = np.reshape(z_sij, (num, num))  # 支持度矩阵
            z_eval, z_evec = np.linalg.eig(z_sij)  # 计算特征值、特征向量
            z_max_eval_idx = np.argsort(z_eval)[-1]  # 最大模特征值对应的索引
            z_V = z_evec[:, z_max_eval_idx:z_max_eval_idx + 1]  # 最大模特征值对应的特征向量
            sum_zV = sum(z_V)
            z_W = np.array([z_V_v / sum_zV for z_V_v in z_V])
            z_pos = np.dot(zz, z_W)[0]  # 融合后的z坐标点

            result = [x_pos, y_pos, z_pos]
    return result


def tdoa_pinv_4_station(station_data):
    '''
    只适应于4个接收站的情况 基本属于obsoletely函数
    :param station_data: 包含接收站ecef坐标、时差等信息的list
    :return: 一对ecef坐标


    '''
    if len(station_data) != 4:
        raise ValueError('Not enough station_data available for pseudo inverse algorithm')

    x0, y0, z0 = station_data[0][0]  # 主站ecef坐标，单位：米
    x1, y1, z1 = station_data[1][0]  # 辅站1 ecef坐标
    x2, y2, z2 = station_data[2][0]  # 辅站2 ecef坐标
    x3, y3, z3 = station_data[3][0]  # 辅站3 ecef坐标

    baseT = station_data[0][1]
    deltaT1 = station_data[1][1] - baseT  # 信号到达辅站1和主站的时差(=t1-t0)，单位：us
    deltaT2 = station_data[2][1] - baseT
    deltaT3 = station_data[3][1] - baseT
    cc = constants.Cair  # 光速， 299792458 / 1.0003 （m/s）
    deltaR1 = deltaT1 * cc * 1e-6  # 信号到达辅站1和主站的距离差，单位：米
    deltaR2 = deltaT2 * cc * 1e-6
    deltaR3 = deltaT3 * cc * 1e-6

    d02 = x0 ** 2 + y0 ** 2 + z0 ** 2
    d12 = x1 ** 2 + y1 ** 2 + z1 ** 2
    d22 = x2 ** 2 + y2 ** 2 + z2 ** 2
    d32 = x3 ** 2 + y3 ** 2 + z3 ** 2

    k1 = 0.5 * (deltaR1 ** 2 + d02 - d12)
    k2 = 0.5 * (deltaR2 ** 2 + d02 - d22)
    k3 = 0.5 * (deltaR3 ** 2 + d02 - d32)

    A = np.array([[x0 - x1, y0 - y1, z0 - z1], [x0 - x2, y0 - y2, z0 - z2], [x0 - x3, y0 - y3, z0 - z3]])
    # print("A =\n", A)
    # print("np.transpose(A) =\n", np.transpose(A))
    # print("np.dot(np.transpose(A), A) = \n", np.dot(np.transpose(A), A)) #矩阵相乘
    # A_inv = np.linalg.inv(A) #逆矩阵
    # print("A_inv =\n", A_inv)
    A_left_pinv = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A))  # 矩阵aa的左伪逆矩阵
    # A_left_pinv = np.dot(np.transpose(A), np.linalg.inv(np.dot(A, np.transpose(A))))  # 矩阵aa的右伪逆矩阵
    # A_left_pinv = np.linalg.inv(A) #逆矩阵
    # print("A_left_pinv =\n", A_left_pinv)

    a11, a12, a13 = A_left_pinv[0]
    a21, a22, a23 = A_left_pinv[1]
    a31, a32, a33 = A_left_pinv[2]

    m1 = a11 * deltaR1 + a12 * deltaR2 + a13 * deltaR3
    m2 = a21 * deltaR1 + a22 * deltaR2 + a23 * deltaR3
    m3 = a31 * deltaR1 + a32 * deltaR2 + a33 * deltaR3
    # print(m1,m2,m3)
    n1 = a11 * k1 + a12 * k2 + a13 * k3
    n2 = a21 * k1 + a22 * k2 + a23 * k3
    n3 = a31 * k1 + a32 * k2 + a33 * k3
    # print(n1,n2,n3)
    a = m1 ** 2 + m2 ** 2 + m3 ** 2 - 1
    b = m1 * n1 - m1 * x0 + m2 * n2 - m2 * y0 + m3 * n3 - m3 * z0
    c = (n1 - x0) ** 2 + (n2 - y0) ** 2 + (n3 - z0) ** 2
    # print(a,b,c)
    ggg = b ** 2 - a * c
    # print(ggg)
    if a <= 0:  # 有唯一解
        r0m_1 = -b / a - math.sqrt(ggg) / a
        r0m_2 = None
    else:
        if b > 0:
            r0m_1 = None  # 无解
            r0m_2 = None
        elif b < 0:
            if ggg > 0:  # 有两个解
                r0m_1 = -b / a + math.sqrt(ggg) / a
                r0m_2 = -b / a - math.sqrt(ggg) / a
            elif ggg == 0:  # 有唯一解
                r0m_1 = -b / a
                r0m_2 = None
            elif ggg < 0:  # 无解
                r0m_1 = None
                r0m_2 = None
    # print(r0m_1,r0m_2)


    xm_1, ym_1, zm_1 = 0, 0, 0
    xm_2, ym_2, zm_2 = 0, 0, 0
    if r0m_1 is not None:
        xm_1 = m1 * r0m_1 + n1
        ym_1 = m2 * r0m_1 + n2
        zm_1 = m3 * r0m_1 + n3
        # print(xm_1,ym_1,zm_1)
        llh1 = geodesy.ecef2llh([xm_1, ym_1, zm_1])
        if llh1[2] < 0:  # 若高度<0，则将结果赋为0
            xm_1, ym_1, zm_1 = 0, 0, 0

    if r0m_2 is not None:
        xm_2 = m1 * r0m_2 + n1
        ym_2 = m2 * r0m_2 + n2
        zm_2 = m3 * r0m_2 + n3
        # print(xm_2,ym_2,zm_2)
        llh2 = geodesy.ecef2llh([xm_2, ym_2, zm_2])
        if llh2[2] < 0:
            xm_2, ym_2, zm_2 = 0, 0, 0
    result = [[xm_1, ym_1, zm_1], [xm_2, ym_2, zm_2]]
    return result


def tdoa_pinv(station_data):
    '''
    适应于接收站大于等于4个的情况
    :param station_data: 包含接收站ecef坐标、时差,时间戳方差等信息的list
    :return: 一对ecef坐标,格式:[[x1,y1,z1], [x2, y2, z2]]
    '''
    if len(station_data) < 4:
        raise ValueError('Not enough station_data available for pseudo inverse algorithm')
    dimension = len(station_data)
    baseT = station_data[0][1]
    position = [receiver for receiver, t, error in station_data]  # 接收站的ecef坐标
    deltaT = [t - baseT for receiver, t, error in station_data]  # 信号到达辅站和主站的时差，单位：us
    # x0, y0, z0 = station_data[0][0]  # 主站ecef坐标，单位：米
    # x1, y1, z1 = station_data[1][0]  # 辅站1 ecef坐标
    # x2, y2, z2 = station_data[2][0]  # 辅站2 ecef坐标
    # x3, y3, z3 = station_data[3][0]  # 辅站3 ecef坐标
    cc = constants.Cair  # 光速， 299792458 / 1.0003 （m/s）
    deltaR = [cc * 1e-6 * time for time in deltaT]  # 信号到达辅站和主站的距离差，单位：米

    distance_square = [p[0] ** 2 + p[1] ** 2 + p[2] ** 2 for p in position]
    # d02 = x0 ** 2 + y0 ** 2 + z0 ** 2
    # d12 = x1 ** 2 + y1 ** 2 + z1 ** 2
    # d22 = x2 ** 2 + y2 ** 2 + z2 ** 2
    # d32 = x3 ** 2 + y3 ** 2 + z3 ** 2

    i = 1
    k = [0]
    while i < dimension:
        k.append(0.5 * (deltaR[i] ** 2 + distance_square[0] - distance_square[i]))
        i += 1
    # k1 = 0.5 * (deltaR1 ** 2 + d02 - d12)
    # k2 = 0.5 * (deltaR2 ** 2 + d02 - d22)
    # k3 = 0.5 * (deltaR3 ** 2 + d02 - d32)

    A = np.array([[position[0][0] - position[1][0], position[0][1] - position[1][1], position[0][2] - position[1][2]],
                  [position[0][0] - position[2][0], position[0][1] - position[2][1], position[0][2] - position[2][2]],
                  [position[0][0] - position[3][0], position[0][1] - position[3][1], position[0][2] - position[3][2]]])
    j = 4
    while j < dimension:
        A = np.row_stack(
            (A, [position[0][0] - position[j][0], position[0][1] - position[j][1], position[0][2] - position[j][2]]))
        j += 1

    # A = np.array([[x0 - x1, y0 - y1, z0 - z1], [x0 - x2, y0 - y2, z0 - z2], [x0 - x3, y0 - y3, z0 - z3]])
    # print("A =\n", A)
    # print("np.transpose(A) =\n", np.transpose(A))
    # print("np.dot(np.transpose(A), A) = \n", np.dot(np.transpose(A), A)) #矩阵相乘
    # A_inv = np.linalg.inv(A) #逆矩阵
    # print("A_inv =\n", A_inv)
    A_left_pinv = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A))  # 矩阵aa的左逆
    # print("A_left_pinv =\n", A_left_pinv)

    # a11, a12, a13 = A_left_pinv[0]
    # a21, a22, a23 = A_left_pinv[1]
    # a31, a32, a33 = A_left_pinv[2]
    deltaR.pop(0)
    m = np.dot(A_left_pinv, deltaR)
    # m1 = a11 * deltaR1 + a12 * deltaR2 + a13 * deltaR3
    # m2 = a21 * deltaR1 + a22 * deltaR2 + a23 * deltaR3
    # m3 = a31 * deltaR1 + a32 * deltaR2 + a33 * deltaR3

    k.pop(0)
    n = np.dot(A_left_pinv, k)
    # n1 = a11 * k[1] + a12 * k[2] + a13 * k[3]
    # n2 = a21 * k[1] + a22 * k[2] + a23 * k[3]
    # n3 = a31 * k[1] + a32 * k[2] + a33 * k[3]

    a = m[0] ** 2 + m[1] ** 2 + m[2] ** 2 - 1
    b = m[0] * n[0] - m[0] * position[0][0] + m[1] * n[1] - m[1] * position[0][1] + m[2] * n[2] - m[2] * position[0][2]
    c = (n[0] - position[0][0]) ** 2 + (n[1] - position[0][1]) ** 2 + (n[2] - position[0][2]) ** 2
    ggg = b ** 2 - a * c

    if a <= 0:  # 有唯一解
        r0m_1 = -b / a - math.sqrt(ggg) / a
        r0m_2 = None
    else:
        if b > 0:
            r0m_1 = None  # 无解
            r0m_2 = None
        elif b < 0:
            if ggg > 0:  # 有两个解
                r0m_1 = -b / a + math.sqrt(ggg) / a
                r0m_2 = -b / a - math.sqrt(ggg) / a
            elif ggg == 0:  # 有唯一解
                r0m_1 = -b / a
                r0m_2 = None
            elif ggg < 0:  # 无解
                r0m_1 = None
                r0m_2 = None

    xm_1, ym_1, zm_1 = 0, 0, 0
    xm_2, ym_2, zm_2 = 0, 0, 0
    if r0m_1 is not None:
        xm_1 = m[0] * r0m_1 + n[0]
        ym_1 = m[1] * r0m_1 + n[1]
        zm_1 = m[2] * r0m_1 + n[2]
        # print(xm_1, ym_1, zm_1)
        llh1 = geodesy.ecef2llh([xm_1, ym_1, zm_1])
        if llh1[2] < 0:  # 若高度<0，则将结果赋为0
            xm_1, ym_1, zm_1 = 0, 0, 0

    if r0m_2 is not None:
        xm_2 = m[0] * r0m_2 + n[0]
        ym_2 = m[1] * r0m_2 + n[1]
        zm_2 = m[2] * r0m_2 + n[2]
        # print(xm_2, ym_2, zm_2)
        llh2 = geodesy.ecef2llh([xm_2, ym_2, zm_2])
        if llh2[2] < 0:
            xm_2, ym_2, zm_2 = 0, 0, 0
    result = [[xm_1, ym_1, zm_1], [xm_2, ym_2, zm_2]]
    return result


if __name__ == "__main__":
    sdata = [[(-2640995.0, 4321663.0, 3863794.0), 0, 0.25], [(-2651352, 4322976, 3855260), 19.9, 0.37],
             [(-2642368, 4327651, 3856180), 22.8, 0.34], [(-2656287, 4317997, 3857399), 23.4, 0.3]]
    sdata = [[(-2640995.0, 4321663.0, 3863794.0), 0.0, 0.3], [(-2656287.0, 4317997.0, 3857399.0), 13.9, 0.25],
             [(-2651352.0, 4322976.0, 3855260.0), 14.7, 0.31], [(-2642368.0, 4327651.0, 3856180.0), 22.0, 0.35]]
    sdata = [[(-2640995.0, 4321663.0, 3863794.0), 0.0, 0.3], [(-2651352.0, 4322976.0, 3855260.0), 18.8, 0.33],
             [(-2656287.0, 4317997.0, 3857399.0), 21.4, 0.25], [(-2642368.0, 4327651.0, 3856180.0), 22.7, 0.37]]
    sdata = [[(-2640995.0, 4321663.0, 3863794.0), 0.0, 0.25], [(-2651352.0, 4322976.0, 3855260.0), 19.9, 0.37],
             [(-2642368.0, 4327651.0, 3856180.0), 22.8, 0.34], [(-2656287.0, 4317997.0, 3857399.0), 23.4, 0.3]]
    sdata = [[(-2656287.0, 4317997.0, 3857399.0), 0.0, 0.25], [(-2651352.0, 4322976.0, 3855260.0), 21.9, 0.25],
             [(-2640995.0, 4321663.0, 3863794.0), 25.3, 0.25],
             [(-2642368.0, 4327651.0, 3856180.0), 46.0, 0.25]]  # 伪逆法无解
    sdata = [[(-2656287.0, 4317997.0, 3857399.0), 0.0, 0.25], [(-2651352.0, 4322976.0, 3855260.0), 22.3, 0.25],
             [(-2640995.0, 4321663.0, 3863794.0), 25.3, 0.25], [(-2642368.0, 4327651.0, 3856180.0), 46.5, 0.25]]
    sdata = [[(-2656287.0, 4317997.0, 3857399.0), 0.0, 0.25], [(-2651352.0, 4322976.0, 3855260.0), 22.3, 0.25],
             [(-2640995.0, 4321663.0, 3863794.0), 25.3, 0.25], [(-2642368.0, 4327651.0, 3856180.0), 46.5, 0.25]]
    sdata = [[(-2640995.0, 4321663.0, 3863794.0), 0.0, 0.25], [(-2642368.0, 4327651.0, 3856180.0), 24.6, 0.25],
             [(-2651352.0, 4322976.0, 3855260.0), 43.7, 0.25], [(-2656287.0, 4317997.0, 3857399.0), 50.9, 0.25]]
    sdata = [[(-2640995.0, 4321663.0, 3863794.0), 0.0, 0.3], [(-2656287.0, 4317997.0, 3857399.0), 13.9, 0.25],
             [(-2651352.0, 4322976.0, 3855260.0), 14.7, 0.31], [(-2642368.0, 4327651.0, 3856180.0), 22.0, 0.35]]
    sdata = [[(-2640995.0, 4321663.0, 3863794.0), 0.0, 0.3], [(-2651352.0, 4322976.0, 3855260.0), 18.8, 0.33],
             [(-2656287.0, 4317997.0, 3857399.0), 21.4, 0.25], [(-2642368.0, 4327651.0, 3856180.0), 22.7, 0.37]]
    sdata = [[(-2640995.0, 4321663.0, 3863794.0), 0.0, 0.25], [(-2651352.0, 4322976.0, 3855260.0), 19.9, 0.37],
             [(-2642368.0, 4327651.0, 3856180.0), 22.8, 0.34], [(-2656287.0, 4317997.0, 3857399.0), 23.4, 0.3]]

    # mubiao = tdoa_pinv_4_station(sdata)
    # print("ecef = ", mubiao)
    # print(" llh = ", geodesy.ecef2llh(mubiao[0]), geodesy.ecef2llh(mubiao[1]))

    measurements = [[(-2640995.0, 4321663.0, 3863794.0), 0, 0.25], [(-2651352, 4322976, 3855260), 19.9, 0.37],
                    [(-2642368, 4327651, 3856180), 22.8, 0.34], [(-2656287, 4317997, 3857399), 23.4, 0.3],
                    [(-2656387, 4318997, 3857499), 26.4, 0.43], [(-2649368, 4327951, 3856980), 29.8, 0.22],
                    [(-2756387, 4318997, 3857499), 36.4, 0.43], [(-2659368, 4327951, 3856980), 39.8, 0.22],
                    [(-2756387, 4322997, 3857499), 46.4, 0.43], [(-2659368, 4327951, 3877980), 49.8, 0.22],
                    [(-2723387, 4322997, 3888499), 56.4, 0.43], [(-2651268, 4344451, 3812980), 59.8, 0.22]]
    # [(-2923387, 4322997, 3812499), 66.4, 0.43], [(-2659968, 4344921, 3812980), 69.8, 0.22],
    # [(-2912217, 4321007, 3623419), 71.4, 0.43], [(-2656368, 3452921, 3722262), 72.8, 0.22],
    # [(-2999217, 4123007, 3623419), 81.4, 0.43], [(-2678968, 3452921, 3801232), 82.8, 0.22],
    # [(-2789017, 4321098, 3789639), 91.4, 0.43], [(-2668978, 3368921, 3900122), 92.8, 0.22]]

    start1 = time.clock()
    result = tdoa_pinv_solve(sdata)
    print("tdoa_pinv_solve计算耗时：", time.clock() - start1)
    print("result num =", len(result), "\necef result =", result)
    print(" llh result =", geodesy.ecef2llh(result[0][0]), geodesy.ecef2llh(result[0][1]))

    start2 = time.clock()
    rel_pos = related_solutions(result)
    print("related_solutions计算耗时：", time.clock() - start2)
    print("相关解个数 =", len(rel_pos), "\n相关解 =", rel_pos)
    # print(geodesy.ecef2llh(rel_pos[0]), geodesy.ecef2llh(rel_pos[1]))
    # dis = geodesy.ecef_distance(rel_pos[0], rel_pos[1])
    # print(dis)
    fus_pos = fusion_solution(rel_pos)
    print("融合解 ecef =", fus_pos)
    print("融合接  llh =", geodesy.ecef2llh(fus_pos))
    print(len(sdata), "个站分组计算融合总耗时：", time.clock() - start1)

    rrr = tdoa_pinv(sdata)
    print("\n", rrr)
    print(geodesy.ecef2llh(rrr[0]), geodesy.ecef2llh(rrr[1]))

    rrr2 = tdoa_pinv_4_station(sdata)
    print(rrr2)
    print(geodesy.ecef2llh(rrr2[0]), geodesy.ecef2llh(rrr2[1]))

    # o_pos = [[[-2656287.0, 4317997.0, 3857399.0], [-2651352.0, 4322976.0, 3855260.0]],
    #              [[-2640995.0, 4321663.0, 3863794.0], [-2642368.0, 4327651.0, 3856180.0]]]
    # r_pos = related_solutions(o_pos, 10e3)
    # print(r_pos)
    # f_pos = fusion_solution(r_pos)
    # print(f_pos)
