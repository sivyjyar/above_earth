import os
import sys
import numpy as np
from rknn.api import RKNN
 
ONNX_MODEL = 'onnx_model.onnx'
RKNN_MODEL = 'netvlad_rknn.rknn'
 
if __name__ == '__main__':
 
    # Create RKNN object
    rknn = RKNN(verbose=True)
 
    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], reorder_channel='0 1 2', target_platform='rv1126',
    quantized_dtype='asymmetric_affine-u8', optimization_level=3,   output_optimize=1)
    print('done')
 
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model  failed!')
        exit(ret)
    print('done')
 
    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build yolov5s failed!')
        exit(ret)
    print('done')
 
    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export yolov5s.rknn failed!')
        exit(ret)
    print('done')
 
    rknn.release()
 
