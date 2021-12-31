import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import *


def get_edge(sum_values, sum_sort_indices, thresh):
    low, high = False, False
    edge = np.zeros(2)

    for i in sum_sort_indices:
        if i < thresh:
            if low:
                if i > edge[0] and sum_values[i] > (0.95*sum_values[int(edge[0])]):
                    edge[0] = i
            else:
                edge[0] = i
                low = True
        elif i > thresh:
            if high:
                if i < edge[1] and sum_values[i] > (0.95*sum_values[int(edge[1])]):
                    edge[1] = i
            else:
                edge[1] = i
                high = True
                
    return edge.astype('int')


def get_function_coordinates(img_path, x_range, y_range, save_excel = False):
    img_gary = cv2.imread(img_path, 0)
    h, w = img_gary.shape
    # 二值化，像素值>250取255，否则取0
    ret, img_binary = cv2.threshold(img_gary, 250, 255, cv2.THRESH_BINARY)
    # cv2.imshow('img_binary', img_binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # 边缘提取
    xgrd = cv2.Sobel(img_binary, cv2.CV_16SC1, 1, 0)
    ygrd = cv2.Sobel(img_binary, cv2.CV_16SC1, 0, 1)
    edge_output = cv2.Canny(xgrd, ygrd, 50, 150)
    # cv2.imshow('img_edge', edge_output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    column_sum = np.sum(edge_output, axis=0)  # 按列求和
    row_sum = np.sum(edge_output, axis=1)  # 按行求和
    
    x1, x2 = get_edge(column_sum, np.argsort(column_sum)[::-1], w/2)
    y1, y2 = get_edge(row_sum, np.argsort(row_sum)[::-1], h/2)
    
    fun_range = edge_output[y1+1:y2, x1+1:x2]
    # 边缘置零, 坐标轴附近可能会出现噪点, 根据个例的情况处理
    fun_range[:,0:2] = 0
    # cv2.imshow('img_fun', fun_range)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    f_h, f_w = fun_range.shape
    min_x, max_x = x_range
    min_y, max_y = y_range
    
    x_axis = np.empty([f_h, f_w])
    y_axis = np.empty([f_w, f_h])
    
    x_value = np.linspace(min_x, max_x, f_w)
    y_value = np.linspace(max_y, min_y, f_h)
    x_axis[:,] = x_value
    y_axis[:,] = y_value
    y_axis = y_axis.T

    x_fc = x_axis.T[fun_range.T==255]
    y_fc = y_axis.T[fun_range.T==255]
    
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.plot(x_fc,y_fc)
    plt.show()
    
    if save_excel:
        xy_data = np.empty([2, x_fc.shape[0]])
        xy_data[0] = x_fc
        xy_data[1] = y_fc
        xy_data = xy_data.T

        data = pd.DataFrame(xy_data)
        data.columns=['x','y']
        csv_path = os.path.splitext(img_path)[0] + '.csv'
        data.to_csv(csv_path)
        # writer = pd
        # writer = pd.ExcelWriter(excel_path)    # 写入Excel文件
        # data.to_excel(writer, 'page_1', float_format='%.5f')    # ‘page_1’是写入excel的sheet名
        # writer.save()
        # writer.close()
        
    return x_fc, y_fc

'''
数据提取参数解释
img_path: 函数图像路径
x_range: 函数图像x轴的取值范围
y_range: 函数图像y轴的取值范围
save_excel: 不需要存excel就设False, 存储的excel文件默认和函数图像文件同名同路径
'''
for da in ['Hangzhou', 'Jinan', '6x6', '6x6-b']:
    for fp in [f'./{da}/only-gcn.png', f'./{da}/only-col.png', f'./{da}/only-ind.png']:
        img_path = fp
        x_range = [0, 200]
        y_range = [0, 1750]
        save_excel = True
        X, Y = get_function_coordinates(img_path, x_range, y_range, save_excel)
        print(X, Y)

        '''
        下面是拟合曲线部分, 如果不需要保存坐标数据, 得到的 X,Y 可以直接进行拟合
        需要拟合的函数和拟合后的可视化还是要自己写的
        '''
        def function1(x,a1,a2,a3,b1,b2,b3,c11,c12,c2,c3):
            return 2*a1 / (np.exp(-(x-b1)/c11)+np.exp((x-b1)/c12)) +\
                    a2 * (b2**2) / (x**2) * np.exp(-((x-b2)/(x*c2))**2) +\
                    a3 * (b3**2) / (x**2) * np.exp(-((x-b3)/(x*c3))**2)
        popt, pcov = curve_fit(function1, X, Y, bounds=([0.,0.,0.,380.,400.,400.,0.,0.,0.,0.], 
                                                    [1.,0.6,0.6,520.,800.,800.,20.,100.,100.,100.]))
        print(popt)
        # print(pcov)

        Y1 = 2*popt[0] / (np.exp(-(X-popt[3])/popt[6])+np.exp((X-popt[3])/popt[7]))
        Y2 = popt[1] * (popt[4]**2) / (X**2) * np.exp(-((X-popt[4])/(X*popt[8]))**2)
        Y3 = popt[2] * (popt[5]**2) / (X**2) * np.exp(-((X-popt[5])/(X*popt[9]))**2)

        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.plot(X, Y, label='data')
        plt.plot(X, function1(X,*popt), label='fit')
        plt.plot(X, Y1, label='T1')
        plt.plot(X, Y2, label='T2')
        plt.plot(X, Y3, label='T3')
        plt.legend(loc='upper right')
        plt.show()
