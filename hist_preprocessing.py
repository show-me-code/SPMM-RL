# -*-coding:utf-8-*-
import scipy.sparse as sparse
from scipy.io import  mmread
import scipy
import numpy as np
from sys import  getsizeof
from ttictoc import tic, toc
import os

#创建一个预处理类，这个类需要把当前的矩阵转换为hist格式，还要负责增广，还要负责标签，以及后续形成数据集
class preprocess():
    def __init__(self, dataset_path_mtx, dataset_path_mat):
        self.sparse_mtx = [] #传入的mtx矩阵
        self.sparse_mat = [] #传入的mat矩阵
        self.sparse_dense = [] #传入已经被转换为标准形势的矩阵
        self.sample_num = 0
        self.dataset_path_mtx = dataset_path_mtx
        self.dataset_path_mat = dataset_path_mat

    def select_pattern_for_mtx(self):
        #选择合适的矩阵存储格式
        #以空格分界，读取第四个内容，如果是pattern就保留，否则删除
        data_list = []
        delete_list = []
        for root, dirs ,files in os.walk(self.dataset_path_mtx):
            #遍历数据集
            for f in files:
                with open(os.path.join(self.dataset_path_mtx, f), 'r') as file_to_delete:
                    line = file_to_delete.readline() #读取第一行
                    pattern = line.split(' ')
                    if pattern[3] == 'pattern':
                        data_list.append(f)
                    else:
                        delete_list.append(f)
        print(data_list)
        with open('data.list', 'a') as txt:
            for i in range(len(delete_list)):
                txt.write(delete_list[i] + '\n')
        for i in data_list:
            os.remove(os.path.join(self.dataset_path_mtx, i))

    def load_mat(self):
        for root, dirs, files in os.walk(self.dataset_path_mat):
            for f in files:
                matrix = scipy.io.loadmat(os.path.join(self.dataset_path_mat, f))
                self.sparse_mat.append(matrix['Problem'][0][0][1]) #存储稀疏矩阵
                self.sample_num += 1
        return self.sparse_mat

    def to_dense(self):
        for i in self.sparse_mat:
            self.sparse_dense.append(i.todense())
        return self.sparse_dense


if __name__ == '__main__':
    '''path = r'/home/drsun/文档/SpMM/SPMM-RL/data/HB/beaflw.mat'
    matrix = scipy.io.loadmat(path)
    print(getsizeof(matrix['Problem'][0][0][1]))
    print(matrix['Problem'][0][0])
    print(np.size(matrix['Problem'][0][0][1].todense()[0]))#获取y轴长度
    m = len(matrix['Problem'][0][0][1].toarray()) #获取x轴长度，其实是个二维数组
    print(m)'''
    p = preprocess(dataset_path_mat=r'/home/drsun/文档/SpMM/SPMM-RL/data/HB/', dataset_path_mtx=' ')
    l = p.load_mat()
    print(l[0])