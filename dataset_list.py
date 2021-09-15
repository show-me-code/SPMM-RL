# -*-coding:utf-8-*-
import os
import shutil
from tqdm import tqdm

#将文件解压之后，遍历所有文件并讲名字改为数字并保存在list里面，最后把所有内容移动到dataset文件夹里
def walk_dataset():
    i = 0
    root = r'/home/drsun/文档/SpMM/SPMM-RL/data/HB'
    path = r'/home/drsun/文档/SpMM/SPMM-RL/data/dataset'
    for root, dirs, files in os.walk(root):
        for f in files:
            #print(f)
            os.rename(os.path.join(root, f), os.path.join(root, str(i)) + '.mtx')
            shutil.move(os.path.join(root, str(i)) + '.mtx', path)
            i += 1


'''def change_name():
    i = 0
    path = r'/home/drsun/文档/RL_optimize/smpv_RL/dnnspmv/data/HB/'
    for file in os.listdir(path):
        print(os.path.join(path, file))
        i += 1'''

#下载的矩阵文件，需要对所有pattern模式的矩阵进行选择
#实现逻辑：以空格为分界，读取第四个内容如果是pattern就保留，否则删除
def select_pattern():
    path = r'/home/drsun/文档/RL_optimize/smpv_RL/dnnspmv/data/dataset'
    root = r'/home/drsun/文档/RL_optimize/smpv_RL/dnnspmv/data'
    data_list = []
    delete_list = []
    for root, dirs, files in os.walk(path):
        for f in files:
            with open(os.path.join(path, f), 'r') as file_to_delete:
                line = file_to_delete.readline()
                #print(line)
                print(f)
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
        os.remove(os.path.join(path, i))


if __name__ == '__main__':
    walk_dataset()
    print('walk finish the data was gathered into dataset/')
    #select_pattern()