import cv2
import numpy as np
import time
# import tensorflow.compat.v1 as tf
# import csv

# import netvlad_tf.net_from_mat as nfm
# import netvlad_tf.nets as nets
# import time

# tf.disable_v2_behavior()

# tf.reset_default_graph()



# inim = cv2.imread('images/1.jpg')
# inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
# bt = inim.astype(np.float32)
# batch = np.expand_dims(bt, axis=0)




# load_tf1('saved-model-builder', batch)





import numpy
import onnxruntime as rt

# sess = rt.InferenceSession("onnx_model.onnx")
# input_name = sess.get_inputs()[0].name
# print(input_name)
# pred_onx = sess.run(None, {input_name: batch})[0]
# print(pred_onx)

st = time.time()

sess = rt.InferenceSession("onnx_model.onnx")
input_name = sess.get_inputs()[0].name

for i in range(1,100):
  inim = cv2.imread(f"dataset/{i}.jpg")
  inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB).astype(np.float32)
  print(i)
  batch = np.expand_dims(inim, axis=0)
  pred_onx = sess.run(None, {input_name: batch})[0]

print(time.time()-st)
