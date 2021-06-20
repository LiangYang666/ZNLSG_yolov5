#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test_nihe.py
# @Time      :2021/6/8 下午1:15
# @Author    :Yangliang
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import norm


def PolynomialRegression(degree):
    return Pipeline([('poly', PolynomialFeatures(degree=degree)),
                     ('std_scaler', StandardScaler()),
                     ('lin_reg', LinearRegression())])

if __name__ == "__main__":
    all_n = 500
    train_n = int(500*0.7)

    x = np.linspace(0, 1, all_n)
    # y = norm.rvs(loc=0, size=500, scale=0.1)
    # y = y + x ** 6
    y = x**4 + 6 + np.sin(x*np.pi)
    # x_points = np.array([0, 1, 2, 3, 4, 5])
    # y_points = np.array([0, 1, 4, 9, 16, 25])

    # xnew = np.linspace(min(x_points), max(x_points), 100)
    poly_reg = PolynomialRegression(degree=10)
    # tck = interpolate.splrep(x_points, y_points)
    poly_reg.fit(x[:train_n, np.newaxis], y[:train_n])
    ynew = poly_reg.predict(x[train_n:, np.newaxis])
    # ynew = interpolate.splev(xnew, tck)

    # plt.scatter(x[:], y[:], 25, "red")
    plt.plot(x, y, 'b', label='real')
    plt.plot(x[train_n:], ynew[:], 'r', label='pred')
    plt.show()