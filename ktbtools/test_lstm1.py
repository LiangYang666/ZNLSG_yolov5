#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test_lstm.py
# @Time      :2021/6/7 下午11:52
# @Author    :Yangliang
import ipdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # matplotlib inline
import math

import torch

from torch import nn
from torch.autograd import Variable
from tqdm import tqdm, trange

if __name__ == "__main__":
    look_back = 50
    dataset = []
    for data in np.arange(0, 3, .01):
        data = math.sin(data * math.pi)
        # data = 5+3*data+data**2
        dataset.append(data)
    dataset = np.array(dataset)
    dataset = dataset.astype('float32')
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    scalar = max_value - min_value
    dataset = list(map(lambda x: x / scalar, dataset))


    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
        return np.array(dataX), np.array(dataY)


    data_X, data_Y = create_dataset(dataset, look_back)

    train_size = int(len(data_X) * 0.7)
    test_size = len(data_X) - train_size
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]


    train_X = train_X.reshape(-1, 1, look_back)
    train_Y = train_Y.reshape(-1, 1, 1)
    test_X = test_X.reshape(-1, 1, look_back)

    train_x = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_Y)
    test_x = torch.from_numpy(test_X)


    class lstm_reg(nn.Module):
        def __init__(self, input_size, hidden_size, output_size=1, num_layers=3):
            super(lstm_reg, self).__init__()

            self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
            self.reg = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # ipdb.set_trace()
            x, _ = self.rnn(x)  # (seq, batch, hidden)
            s, b, h = x.shape
            x = x.view(s * b, h)
            x = self.reg(x)
            x = x.view(s, b, -1)
            return x


    net = lstm_reg(look_back, 40)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    for e in tqdm(range(30)):
        optimizer.zero_grad()
        var_x = Variable(train_x)
        var_y = Variable(train_y)
        out = net(var_x)
        loss = criterion(out, var_y)
        loss.backward()
        optimizer.step()
        if (e + 1) % 100 == 0:
            # print('Epoch: {}, Loss: {:.10f}'.format(e + 1, loss.data[0]))
            print('Epoch: {}, Loss: {:.10f}'.format(e + 1, loss.item()))
    net = net.eval()
    data_X = data_X.reshape(-1, 1, look_back)
    data_X = torch.from_numpy(data_X)
    var_data = Variable(data_X)
    pred_test = net(data_X)
    pred_test = pred_test.view(-1).data.numpy()
    plt.plot(pred_test, 'r', label='prediction')
    plt.plot(dataset[look_back:], 'b', label='real')
    plt.legend(loc='best')