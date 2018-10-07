#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/7 20:16
# @Author  : jxyxyj@gmail.com
# @Site    : 
# @File    : Keras_Mnist_Hello.py.py
# @Software: PyCharm

import numpy as np


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from keras.optimizers import SGD


def main():
    #准备数据
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    print(x_train.shape)
    print(x_test.shape)

    x_train = x_train.reshape(60000,784)
    x_test = x_test.reshape(10000,784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    y_train = np_utils.to_categorical(y_train,10)
    y_test = np_utils.to_categorical(y_test,10)

    #建网络
    model = Sequential()
    model.add(Dense(input_dim=784,units=500,activation="sigmoid"))
    model.add(Dense(units=500,activation="sigmoid"))
    model.add(Dense(units=500,activation="sigmoid"))
    # model.add(Dense(units=10,activation="sigmoid"))
    model.add(Dense(units=10, activation="softmax"))
    model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.1),metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=100,epochs=20)
    result = model.evaluate(x_test,y_test)
    print("正确率",end='')
    print(result[1])


if __name__ == '__main__':
    # print("hello")
    main()