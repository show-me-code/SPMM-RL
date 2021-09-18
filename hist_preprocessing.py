# -*-coding:utf-8-*-
import scipy.sparse as sparse
from scipy.io import  mmread
import scipy
import numpy as np
import numpy.matlib as matlib
from sys import  getsizeof
from ttictoc import tic, toc
import os

#创建一个预处理类，这个类需要把当前的矩阵转换为hist格式，还要负责增广，还要负责标签，以及后续形成数据集
class preprocess():
    def __init__(self, dataset_path_mtx, dataset_path_mat):
        self.sparse_mtx = [] #传入的mtx矩阵
        self.sparse_mat = [] #传入的mat矩阵
        self.sparse_dense = [] #传入已经被转换为标准形势的矩阵
        self.sparse_hist = []
        self.sample_num = 0
        self.dataset_path_mtx = dataset_path_mtx
        self.dataset_path_mat = dataset_path_mat

    def select_pattern_for_mtx(self):
        #选择合适的矩阵存储格式，读取mtx格式
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
        #加载mat类型的矩阵并将其载入内存
        #load完还是scipy sparse格式
        for root, dirs, files in os.walk(self.dataset_path_mat):
            for f in files:
                matrix = scipy.io.loadmat(os.path.join(self.dataset_path_mat, f))
                self.sparse_mat.append(matrix['Problem'][0][0][1]) #存储稀疏矩阵
                self.sample_num += 1
        return self.sparse_mat

    def to_dense(self):
        #在完成load_mat之后，使用这一函数将其转化为dense模式并换位ndarray
        for i in self.sparse_mat:
            dense = i.todense()
            narr = dense.getA()
            self.sparse_dense.append(narr)
        return self.sparse_dense

    #创建行的histogram represention 测试通过，可能要配合col填入双通道
    #要测试一下单通道怎么样
    def to_hist_row(self, r, BINS):
        for i in range(len(self.sparse_dense)):
            hist_matrix = np.zeros((r, BINS))
            height, width = np.shape(self.sparse_dense[i])[0], np.shape(self.sparse_dense[i])[1]
            scale_ratio = int(height/r)
            max_dim = max(height, width)
            temp = sparse.coo_matrix(self.sparse_dense[i]) #用coo格式找出来所有非0行列
            col = temp.col #非0元素索引的列
            row = temp.row #非0元素索引的行
            enrty = temp.data
            for i in range(len(enrty)):
                row_r = row[i]/scale_ratio
                bin = BINS* abs(row[i] - col[i]) / max_dim
                hist_matrix[row_r][bin] += 1
        return hist_matrix

    #可能之后要填入双通道
    def to_hist_col(self, r, BINS):
        for i in range(len(self.sparse_dense)):
            hist_matrix = np.zeros((r, BINS))
            height, width = np.shape(self.sparse_dense[i])[0], np.shape(self.sparse_dense[i])[1]
            scale_ratio = int(height/r)
            max_dim = max(height, width)
            temp = sparse.coo_matrix(self.sparse_dense[i]) #用coo格式找出来所有非0行列
            col = temp.col #非0元素索引的列
            row = temp.row #非0元素索引的行
            enrty = temp.data
            for i in range(len(enrty)):
                row_r = col[i]/scale_ratio
                bin = BINS* abs(row[i] - col[i]) / max_dim
                hist_matrix[row_r][bin] += 1
        return hist_matrix

    #test check
    def test_hist(self, A, r, BINS):
        hist_matrix = np.zeros((r, BINS))
        height, width = np.shape(A)[0], np.shape(A)[1]
        scale_ratio = int(height / r)
        max_dim = max(height, width)
        #A = A.T
        temp = sparse.coo_matrix(A)  # 用coo格式找出来所有非0行列
        col = temp.col  # 非0元素索引的列
        row = temp.row  # 非0元素索引的行
        enrty = temp.data
        for i in range(len(enrty)):
            row_r = int(col[i] / scale_ratio)
            print(row_r)
            bin = int(BINS * abs(col[i] - row[i]) / max_dim)
            print(bin)
            hist_matrix[row_r][bin] += 1
        return hist_matrix




if __name__ == '__main__':
    '''path = r'/home/drsun/文档/SpMM/SPMM-RL/data/HB/beaflw.mat'
    matrix = scipy.io.loadmat(path)
    print(getsizeof(matrix['Problem'][0][0][1]))
    print(matrix['Problem'][0][0])
    print(np.size(matrix['Problem'][0][0][1].todense()[0]))#获取y轴长度
    m = len(matrix['Problem'][0][0][1].toarray()) #获取x轴长度，其实是个二维数组
    print(m)'''
    p = preprocess(dataset_path_mat=r'/home/drsun/文档/SpMM/SPMM-RL/data/HB/', dataset_path_mtx=' ')
    '''l = p.load_mat()
    #p.to_hist( r=2, BINS=2)
    print(type(l[0]))
    l = p.to_dense()
    print(np.shape(l[0]))
    a = sparse.coo_matrix(l[0]) #转换为特定格式
    print(a)
    print(len(a.col)) #找到特定格式的列
    print(len(a.row)) #找到特定格式的行
    print(len(a.data)) #找到对应的数据
    #print(l[0].tocoo())
    #print(np.size(l[0][0]))
    #print(len(l[0]))
    #print(np.shape(l[0]))
    #print(l)
    '''

    A = [[45, 0, 0, 0, 0, 0, 0, 0],
         [0, -25, 0, 0, 0, 0, 0, 0],
         [0, 0, 89, 37, 0, 0, 0, 0],
         [0, 0, 43, 94, 0, 0, 0, 0],
         [77, 0, 0, 0, 0, 15, 0, 0],
         [0, 0, 0, 0, 36, 78, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 23],
         [0, 0, 0, 17, 0, 0, 11, 0]]
    A = np.array(A)
    B = p.test_hist(A=A, r=4, BINS=4)
    print(B)