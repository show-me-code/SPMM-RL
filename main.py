
import scipy.sparse as sparse
from scipy.io import mmread, mmwrite, mminfo
import numpy as np
from sys import getsizeof
from ttictoc import tic, toc
import os
#也许可以通过计算稀疏矩阵乘的时间来进行判定，我们发现存储不同格式的大小并不会带来什么变化
def load_mtx_directly():
    dok, coo, dense, bsr, csr, lil, csc = 0, 0, 0, 0, 0, 0, 0
    tic()
    for root, dirs, file in os.walk('dataset'):
        for f in file:
            print(f)
            matrix = mmread('dataset/'+ f)
            y_size = np.size(mmread('dataset/' + f).toarray()[1])
            mult = np.random.randn(y_size)
            matrix2 = matrix.todok()

            tic()
            matrix2.dot(mult)
            dok += toc()

            matrix2 = matrix.tocoo()
            tic()
            matrix2.dot(mult)
            coo += toc()

            matrix2.todense()
            tic()
            matrix2.dot(mult)
            dense += toc()

            matrix2.tobsr()
            tic()
            matrix2.dot(mult)
            bsr += toc()

            matrix2.tocsr()
            tic()
            matrix2.dot(mult)
            csr += toc()

            matrix2.tocsc()
            tic()
            matrix2.dot(mult)
            csc += toc()

            matrix2.tolil()
            tic()
            matrix2.dot(mult)
            lil += toc()
    mark = toc()
    print('mark time', mark)
    #mmwrite('dok0.mtx', matrix2)
    print('dok', dok, '\ncoo', coo, '\ndense', dense, '\nbsr', bsr, '\ncsr', csr, '\ncsc', csc, '\nlil', lil)

def gen_sparse_matrix():
    #a = np.eyes((114514, 114514))
    a = sparse.eye(1145).toarray()
    print(getsizeof(a), type(a))
    spa = sparse.coo_matrix(a)
    print(spa)
    print('coo', getsizeof(spa.toarray()), type(spa))
    spb = sparse.csr_matrix(a)
    print('csr', getsizeof(spb.toarray()))
    spc = sparse.dia_matrix(a)
    print(spc)
    print('dia', getsizeof(spc.toarray()))

if __name__ == '__main__':
    load_mtx_directly()
