# coding=utf-8
import struct
from time import sleep
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np

# 数据的准备
# 注意将目录改成你的数据目录
# 处理训练图片
def getImage ():
    imgs = np.zeros([60000,784],int)
    print("提取训练图片")
    for i in tqdm(range(60000)):
        img0 = cv2.imread(r"D:\data\mvist\testIM"+str(i)+".jpg",0)
        for row in range(28):
            for col in range(28):
                if img0[row,col]>=127 :
                    img0[row,col]=1
                else:
                    img0[row,col]=0
                imgs[i,28*row+col] = img0[row,col]

    return imgs

#处理测试图片
def getTestImage ():
    imgs = np.zeros([10000, 784], int)
    print("提取测试图片")
    for i in tqdm(range(10000)):
        img0 = cv2.imread(r"D:\data\test\te" + str(i) + ".jpg", 0)
        for row in range(28):
            for col in range(28):
                if img0[row, col] >= 127:
                    img0[row, col] = 1
                else:
                    img0[row, col] = 0
                imgs[i, 28 * row + col] = img0[row, col]

    return imgs

# 得到训练集标签
def getLabels():
    with open(r"D:\data\train-labels.idx1-ubyte","rb") as f1:
        buf1 = f1.read()

    index = 0
    magic,num = struct.unpack_from(">II",buf1,0)
    index += struct.calcsize(">II")
    labs = []
    labs = struct.unpack_from('>'+str(num)+'B',buf1,index)
    return labs

# 得到测试集标签
def getTestLabels():
    with open(r"D:\data\test\t10k-labels.idx1-ubyte","rb") as f1:
        buf1 = f1.read()

    index = 0
    magic,num = struct.unpack_from(">II",buf1,0)
    index += struct.calcsize(">II")
    labs = []
    labs = struct.unpack_from('>'+str(num)+'B',buf1,index)
    return labs




# 得到feature为1 的个数
def getProbabilityOfFeature(train_images,train_labels,number,tlc):

    # 统计列表里的重复值, 把number的index全部取出，放到list_index里
    list_index = []
    flag = 0

    #print("各个label有多少个"+str(tlc))
    for i in range(tlc):
        sec = flag
        flag = train_labels[flag:].index(number)
        list_index.append(flag+sec)
        #flag = flag+sec+1
        flag = list_index[-1:][0] +1
    # print("list_index ")
    # print(list_index)

    # 统计list_index所指的多个向量 ，第col个feature的和
    num_of_one = np.zeros([784],int)
    for i in range(tlc):
        num_of_one +=train_images[list_index[i],:]
   # print("num_of_one")
  #  print(num_of_one)
    return num_of_one



# # 处理features是number的概率
# def getProbability(features,number):
#     for i in range(784):
#         t1 = features[i]


# 得到这个features最可能的数值
def getResult(features,p_num,p_feature_num):
    pro = 0
    #print("计算可能的值")
    for i in range(10):

        p_f_total = 1
        for j in range(784):
            if features[j] == 1 :
                p_f_total *= p_feature_num[i,j]
            else:
                #p_f_total *= (1-p_feature_num[i,j])
                p_f_total *= (1 - p_feature_num[i, j])
        #print("p_f_total is %e" %p_f_total)
        temp = p_num[i]*p_f_total
        #print("可能性为 %e"%temp)

        if pro <temp:
            pro = temp
            result =i
            #print("有更大的")
    #print("返回的值为 %d"%result)
    return result



# 得到错误率，也就是让 真实label和预测的去比
def getErrorRate(test_images,test_labels,p_num,p_feature_num):

    print(p_feature_num[2:])
    error_num=0
    print("--------开始计算错误率--------")
    for i in tqdm(range(10000)):
        rr = getResult(test_images[i, :], p_num, p_feature_num)
        if test_labels[i] != rr:
            error_num +=1
            print("*****------识别错误------****，将",end='')
            print(test_labels[i],end='')
            print("识别为",end='')
            print(rr)
    print("--------正确率率计算完成--------")
    print("正确数目为 %d" %(10000-error_num))
    print("错误数目为 %d" %error_num)
    return error_num/10000




def main():
    print ('开始')
    # 训练图
    train_images = getImage()
    print("----get train_images----")
    train_labels = getLabels()
    print("============TRAINING_LABELS",end='')
    print(train_labels)

    print("get train_labels----")
    test_images = getTestImage()
    print("get test_images----")
    test_labels = getTestLabels()
    print("get test_labels---")
    print("=============TEST_LABELS",end='')
    print(test_labels)
    #  得到统计量p_num 记录了 0-9 的出现的概率

    p_num = [0]*10
    for i in tqdm(range(10)):
        temp = (test_labels.count(i)) / len(test_labels)
        p_num[i] = (temp)
        print(i,end='')
        print('的概率为',end='')
        print(temp)


    print("-------先验概率计算完成-------")

    print("-------开始计算条件概率-------")

    p_feature_num = np.zeros([10, 784])
    for i in tqdm(range(10)):
        cont = train_labels.count(i)

        # 加1 防止个数为0

        pof = getProbabilityOfFeature(train_images, train_labels, i,  cont)
        #pof = np.array(pof)
        pof = pof +1
        #pof = np.array(pof)
        print("pof")
        print(pof)
        #sssssss = input("please")
        #p_f_n = np.zeros([1,784])
        #p_f_n = (pof) / (cont + 1)
        p_f_n = pof /(cont + 1)
        #p_f_n = np.array(p_f_n)

        print("pfn")
        print(p_f_n)
        #input("please")
        p_feature_num[i,:] = p_f_n + p_feature_num[i,:]
        print('i:', end='')
        print(i, end='')
        print("条件概率")
        print(p_feature_num[i,:])
        #input("please ")
    print(p_feature_num)
    #ssss = input("please ")
    print("-------条件概率矩阵计算完成-------")

    print("计算正确率")
    error_rate = getErrorRate(test_images,test_labels,p_num,p_feature_num)
    print('正确率是 %f ' % ((1-error_rate)*100),end='')
    print('%')




if __name__ == '__main__':
     main()
