import os
import time

test_path = ['guazi',
             'guazi_resize',
             'head',
             'in',
             'mobilephone',
             'out',
             'park',
             'poc',
             'tail',
             'toll',
             'truck']
time_start = time.time()
for i in test_path:
    os.system('CUDA_VISIBLE_DEVICES=0,1,2,3 python test_ISDAloss.py resnet18 '+i)

time_end = time.time()
print('time cost: ', time_end - time_start, 's')
