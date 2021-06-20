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
    look_back = 10
    dataset = []
    for data in np.arange(1, 3, .01):
        data = math.sin(data * math.pi)
        # data = 5 + 3 * data + data ** 2
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
    # ipdb.set_trace()

    train_size = int(len(data_X) * 0.7)
    test_size = len(data_X) - train_size
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]

    # train_X = train_X.reshape(-1, 1, look_back)
    # train_Y = train_Y.reshape(-1, 1, 1)
    # test_X = test_X.reshape(-1, 1, look_back)
    #
    # train_x = torch.from_numpy(train_X)
    # train_y = torch.from_numpy(train_Y)
    # test_x = torch.from_numpy(test_X)


    class lstm_reg(nn.Module):
        def __init__(self, input_size=1, hidden_size=40, output_size=1, num_layers=1):
            super(lstm_reg, self).__init__()
            self.hidden_layer_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
            self.linear = nn.Linear(hidden_size, output_size)
            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                                torch.zeros(1, 1, self.hidden_layer_size))

        def forward(self, input_seq):
            # ipdb.set_trace()
            lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)     # (seq, batch, hidden)
            predictions = self.linear(lstm_out.view(len(input_seq), -1))
            return predictions[-1]

    model = lstm_reg()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for e in tqdm(range(100)):
        # var_x = Variable(train_x)
        # var_y = Variable(train_y)
        epoch_loss = 0
        for i in range(len(train_X)):
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            optimizer.zero_grad()
            # var_x = train_x[i, :, :].unsqueeze(0)
            # var_y = train_y[i, :, :].unsqueeze(0)
            var_x = torch.from_numpy(train_X[i])
            var_y = torch.Tensor([train_Y[i]])
            out = model(var_x)
            loss = criterion(out[-1], var_y)
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        if (e + 1) % 10 == 0:
            # print('Epoch: {}, Loss: {:.10f}'.format(e + 1, loss.data[0]))
            print('Epoch: {}, Loss: {:.10f}'.format(e + 1, epoch_loss.item()))
    model = model.eval()
    # data_X = data_X.reshape(-1, 1, look_back)
    # data_X = torch.from_numpy(data_X)
    # var_data = Variable(data_X)
    pred_data = torch.from_numpy(test_X[0].copy())
    for i in range(len(test_X)):
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        pred_test = model(pred_data[i: (i+1) * look_back])
        # pred_test = model(torch.from_numpy(test_X[i]))
        pred_data = torch.cat((pred_data, pred_test))

    pred_test = pred_data.view(-1).detach().numpy()
    plt.plot(dataset[:], 'b', label='real')
    plt.plot(list(range(len(dataset)-len(pred_test)+look_back, len(dataset))),pred_test[look_back:], 'r', label='prediction')
    plt.legend(loc='best')
