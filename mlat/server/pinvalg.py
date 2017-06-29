import numpy as np
import math
import itertools

from mlat import geodesy, constants

'''
伪逆法求解时差定位的非线性方程组
'''


def tdoa_pinv_solve(measurements, altitude=10000, altitude_error=None):
    '''

    :param measurements:
    :param altitude:
    :param altitude_error:
    :return: 多组解组成的list。需要进一步处理：选取相关解、数据融合
    '''
    if len(measurements) < 4:
        raise ValueError('Not enough measurements available for pseudo inverse algorithm')
    base_timestamp = measurements[0][1]
    pseudorange_data = [[receiver,  # receiver.position, #实际传入的参数可能是类对象，需要receiver.positon
                         timestamp - base_timestamp,
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
        # print("rr=",rr)
        i += 1
        if not rr:
            continue
        elif len(rr) == 2:
            result.append(rr[0])
            result.append(rr[1])
        else:
            result.append(rr[0])
    # print(result)
    return result


def related_solutions(solution):
    '''

    :param solution: tdoa_pinv_solve求得的解的list
    :return: 相关解
    '''
    return solution


def fusion_solution(related_solu):
    '''

    :param related_solu: 相关解
    :return: 相关解经过数据融合之后得到的唯一一组解
    '''

    return


def tdoa_pinv_4_station(station_data):
    '''
    只适应于4个接收站的情况 基本属于obsoletely函数
    :param station_data: 包含接收站ecef坐标、时差等信息的list
    :return: ecef position


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
    A_left_pinv = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A))  # 矩阵aa的左逆
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

    result = []
    xm_1, ym_1, zm_1 = 0, 0, 0
    xm_2, ym_2, zm_2 = 0, 0, 0
    if r0m_1 is not None:
        xm_1 = m1 * r0m_1 + n1
        ym_1 = m2 * r0m_1 + n2
        zm_1 = m3 * r0m_1 + n3
        # print(xm_1,ym_1,zm_1)
        # llh1 = geodesy.ecef2llh([xm_1, ym_1, zm_1])
        # if llh1[2] >= 0: #高度>=0
        #     result.append([xm_1, ym_1, zm_1])

    if r0m_2 is not None:
        xm_2 = m1 * r0m_2 + n1
        ym_2 = m2 * r0m_2 + n2
        zm_2 = m3 * r0m_2 + n3
        # print(xm_2,ym_2,zm_2)
        # llh2 = geodesy.ecef2llh([xm_2, ym_2, zm_2])
        # if llh2[2] >= 0:
        #     result.append([xm_2, ym_2, zm_2])
    result.append([xm_1, ym_1, zm_1])
    result.append([xm_2, ym_2, zm_2])
    return result


def tdoa_pinv(station_data):
    '''
    适应于接收站大于等于4个的情况
    :param station_data: 包含接收站ecef坐标、时差等信息的list
    :return: ecef position
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
    m = np.dot(A_left_pinv, deltaR)  # 将数组转换成矩阵
    # m1 = a11 * deltaR1 + a12 * deltaR2 + a13 * deltaR3
    # m2 = a21 * deltaR1 + a22 * deltaR2 + a23 * deltaR3
    # m3 = a31 * deltaR1 + a32 * deltaR2 + a33 * deltaR3
    # print(m1,m2,m3)

    k.pop(0)
    n = np.dot(A_left_pinv, k)
    # n1 = a11 * k[1] + a12 * k[2] + a13 * k[3]
    # n2 = a21 * k[1] + a22 * k[2] + a23 * k[3]
    # n3 = a31 * k[1] + a32 * k[2] + a33 * k[3]
    # print(n1,n2,n3)
    a = m[0] ** 2 + m[1] ** 2 + m[2] ** 2 - 1
    b = m[0] * n[0] - m[0] * position[0][0] + m[1] * n[1] - m[1] * position[0][1] + m[2] * n[2] - m[2] * position[0][2]
    c = (n[0] - position[0][0]) ** 2 + (n[1] - position[0][1]) ** 2 + (n[2] - position[0][2]) ** 2
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

    result = []
    xm_1, ym_1, zm_1 = 0, 0, 0
    xm_2, ym_2, zm_2 = 0, 0, 0
    if r0m_1 is not None:
        xm_1 = m[0] * r0m_1 + n[0]
        ym_1 = m[1] * r0m_1 + n[1]
        zm_1 = m[2] * r0m_1 + n[2]
        # print(xm_1,ym_1,zm_1)
        # llh1 = geodesy.ecef2llh([xm_1, ym_1, zm_1])
        # if llh1[2] >= 0: #高度>=0
        #     result.append([xm_1, ym_1, zm_1])

    if r0m_2 is not None:
        xm_2 = m[0] * r0m_2 + n[0]
        ym_2 = m[1] * r0m_2 + n[1]
        zm_2 = m[2] * r0m_2 + n[2]
        # print(xm_2,ym_2,zm_2)
        # llh2 = geodesy.ecef2llh([xm_2, ym_2, zm_2])
        # if llh2[2] >= 0:
        #     result.append([xm_2, ym_2, zm_2])
    result.append([xm_1, ym_1, zm_1])
    result.append([xm_2, ym_2, zm_2])
    return result


if __name__ == "__main__":
    sdata = [[(-2640995.0, 4321663.0, 3863794.0), 0, 0.25], [(-2651352, 4322976, 3855260), 19.9, 0.37],
             [(-2642368, 4327651, 3856180), 22.8, 0.34], [(-2656287, 4317997, 3857399), 23.4, 0.3]]
    mubiao = tdoa_pinv_4_station(sdata)
    print("ecef = ", mubiao)
    print(" llh = ", geodesy.ecef2llh(mubiao[0]), geodesy.ecef2llh(mubiao[1]))

    measurements = [[(-2640995.0, 4321663.0, 3863794.0), 0, 0.25], [(-2651352, 4322976, 3855260), 19.9, 0.37],
                    [(-2642368, 4327651, 3856180), 22.8, 0.34], [(-2656287, 4317997, 3857399), 23.4, 0.3],
                    [(-2656387, 4318997, 3857499), 26.4, 0.43], [(-2649368, 4327951, 3856980), 29.8, 0.22]]

    result = tdoa_pinv_solve(measurements)
    print(result)
    # print(len(result))
    print(geodesy.ecef2llh(result[0]), geodesy.ecef2llh(result[1]))

    rrr = tdoa_pinv(measurements)
    print(rrr)
    print(geodesy.ecef2llh(rrr[0]), geodesy.ecef2llh(rrr[1]))
