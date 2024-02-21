import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import csv
tf.disable_v2_behavior()

tf.reset_default_graph()

image_batch = tf.placeholder(
        dtype=tf.float32, shape=[None, None, None, 3])

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph
 
graph = load_pb('output_graph.pb')       
input = graph.get_tensor_by_name('input:0')
output = graph.get_tensor_by_name('output:0')


        
inim = cv2.imread('images/1.jpg')
inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)

batch = np.expand_dims(inim, axis=0)
result = sess.run(net_out, feed_dict={input: batch})  # расчет графов
