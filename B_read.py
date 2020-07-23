import sys
import os 
import torch
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_x_y_labels(file):       # 获取单独一对x，y的标签值
    raw_label, _ = [i for i in file.split('.')]
    #raw_label1 = os.path.splitext(sfile)[0]       #另外一种获取文件前缀的方法
    x, y = [i for i in raw_label.split('y')]
    x = float(x[1:])
    y = float(y)
    return x, y

def create_one_B_matrix(B_list):    # 生成一个磁感应强度分布矩阵（借助仿真数据）
    B_matrix = np.zeros((31,31))    # 规定 磁感应强度分布矩阵 B_matrix 行代表y，列代表x
    for i in range(31):
        B_matrix[:,i] = B_list[i*31:(i+1)*31]
    B_list.clear()
    return B_matrix

def revised_by_zero(Matrix_list,labels):  #将所有情况的磁场分布均减去无偏移时的分布以修正数据
    align_index = labels.tolist().index([0,0])  # 找到正中心无偏移时的磁场分布
    Matrix_list_new = Matrix_list - Matrix_list[align_index]
    return Matrix_list_new

def create_one_revised_mesh(i,Matrix_list_revised,labels):    # 绘制一个经无偏移磁分布修正的准三维图像
    _, ax = plt.subplots()
    ax.set_title('x:'+str(labels.tolist()[i][0])+'  y:'+str(labels.tolist()[i][1]))
    x_or_y = np.arange(-45, 48, 3) 
    ax.pcolormesh(x_or_y, x_or_y, Matrix_list_revised[i].numpy())
    plt.show()

def generate_sensor_location_list():  # 生成传感器位置序列  该函数里自己随意修改生成方法，这里只是想封装起来
    x = np.hstack((np.zeros(6,dtype=int)  
                  ,np.arange(6,31,6)  
                  ,np.ones(5,dtype=int)*30
                  ,np.arange(24,1,-6))).reshape(20,1)
    y = np.hstack((np.arange(0,31,6)
                  ,np.ones(5,dtype=int)*30
                  ,np.arange(24,-1,-6)
                  ,np.zeros(4,dtype=int))).reshape(20,1)
    location = np.hstack((x, y)) 
    return location

def get_train_inf(Matrix_list_revised):    # 获得待训练向量
    location = generate_sensor_location_list()
    train_B_list = torch.empty(len(Matrix_list_revised),len(location))
    for i in range(len(Matrix_list_revised)):
        for x, y in location:
            j = location.tolist().index([x, y])
            train_B_list[i][j] = Matrix_list_revised.numpy()[i][y][x]
    return train_B_list

def get_labels_and_trainlist():

    label = [[],[]]   #存入副边偏移坐标值(x,y)标签，数据 torch.tensor:shape 161×2
    B_temp = []     #临时存放磁感应强度的列表
    Matrix_list = []   #存入所有仿真得到的磁感应强度分布数据，数据 torch.tensor:shape 161×31×31 
    
    filenamelist = os.listdir(os.path.dirname(os.path.abspath(__file__))+'/txt')
    for sfile in filenamelist:
        label_x, label_y = get_x_y_labels(sfile)
        label[0].append(label_x)
        label[1].append(label_y)

        with open(os.path.dirname(os.path.abspath(__file__))+'/txt/'+sfile,"r") as f:
            line = f.readline()
            line = f.readline()
            while line:           
                line = f.readline() 
                try:
                    lx, ly, lz, Br = [i for i in line.split()]
                    #B = Br.split('e')[0] #单位：mTsl（毫特斯拉）
                    B_temp.append(float(Br.split('e')[0]))
                except ValueError:
                    pass
            Matrix_list.append(create_one_B_matrix(B_temp))
            f.close() 

    labels = torch.tensor(label,dtype=torch.float32).T
    Matrix_list = torch.tensor(Matrix_list)
    Matrix_list_revised = revised_by_zero(Matrix_list,labels)
    #create_one_revised_mesh(41,Matrix_list_revised,labels)

    Train_B_list = get_train_inf(Matrix_list_revised)

    return Train_B_list, labels



if __name__=="__main__":

    label = [[],[]]   #存入副边偏移坐标值(x,y)标签，数据 torch.tensor:shape 161×2
    B_temp = []     #临时存放磁感应强度的列表
    Matrix_list = []   #存入所有仿真得到的磁感应强度分布数据，数据 torch.tensor:shape 161×31×31 

    filenamelist = os.listdir(os.path.dirname(os.path.abspath(__file__))+'/txt')
    for sfile in filenamelist:
        label_x, label_y = get_x_y_labels(sfile)
        label[0].append(label_x)
        label[1].append(label_y)

        with open(os.path.dirname(os.path.abspath(__file__))+'/txt/'+sfile,"r") as f:
            line = f.readline()
            line = f.readline()
            while line:           
                line = f.readline() 
                try:
                    lx, ly, lz, Br = [i for i in line.split()]
                    #B = Br.split('e')[0] #单位：mTsl（毫特斯拉）
                    B_temp.append(float(Br.split('e')[0]))
                except ValueError:
                    pass
            Matrix_list.append(create_one_B_matrix(B_temp))
            f.close() 

    labels = torch.tensor(label,dtype=torch.float32).T
    Matrix_list = torch.tensor(Matrix_list)
    Matrix_list_revised = revised_by_zero(Matrix_list,labels)
    #create_one_revised_mesh(41,Matrix_list_revised,labels)

    Train_B_list = get_train_inf(Matrix_list_revised)
    print(Train_B_list[60])
