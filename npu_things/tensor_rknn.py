from rknn.api import RKNN  
import cv2
import tensorflow.compat.v1 as tf
import csv

import netvlad_tf.net_from_mat as nfm
import netvlad_tf.nets as nets
import time
import numpy as np

tf.disable_v2_behavior()

tf.reset_default_graph()


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph



image_batch = tf.placeholder(
        dtype=tf.float32, shape=[None, None, None, 3])

net_out = nets.vgg16NetvladPca(image_batch)
saver = tf.train.Saver() #создает место для хранения переменных

with tf.Session() as sess:

        saver.restore(sess, nets.defaultCheckpoint())

        inim = cv2.imread('images/1.jpg')
        inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)



        st = time.time()
        batch = np.expand_dims(inim, axis=0)
 

        # result = sess.run(net_out, feed_dict={image_batch: batch})  # расчет графов


        frozen_graph = freeze_session(sess)
        tf.train.write_graph(frozen_graph, "model", "tf_model.pb", as_text=False)


# #
# INPUT_SIZE = 64
# platform = 'rk3588'
#
# if __name__ == '__main__':
#     # Create a RKNN execution object
#     rknn = RKNN()
#          # Configure the model input for the pre-processing of the data input by NPU
#          # channel_mean_value = '0 0 0 255', then the RGB data will be converted as follows when the model is reasoning.
#          # (R - 0) / 255, (g - 0) / 255, (b - 0) / 255. When reasoning, the RKNN model will automatically do mean and normalize.
#          # REORDER_CHANNEL = '0 1 2' Used to specify whether to adjust the image channel order, set to 0 1 2, press the input image channel order, not adjust
#          # REORDER_CHANNEL = '2 1 0' Indicates switches 0 and 2 channels, and if the input is RGB, it will be adjusted to BGR. If it is BGR, it will be adjusted to RGB
#          # Image channel order does not adjust
#     rknn.config(target_platform=platform)
#
#          #   Tensorflow model
#          # TF_PB = 'DIGITAL_GITATION.PB' Specified Tensorflow Model for Terring
#          # INPUTS Specify the input node in the model
#          # OUTPUTS Specify the output node in the model
#          # input_size_list Specify the size of the model input
#     print('--> Loading model')
#     rknn.load_tensorflow(tf_pb='model/tf_model.pb',
#                          inputs=['input'],
#                          outputs=['output'],
#                          input_size_list=[[1, INPUT_SIZE, INPUT_SIZE, 3]])
#     print('done')
#
#          # Create a parsing PB model
#          # do_quantization = false Specifies not to quantify
#          # Quantization reduces the volume of the model and enhances the calculation speed, but will have a loss of accuracy
#     print('--> Building model')
#     rknn.build(do_quantization=False)
#     print('done')
#
#          # Export Save RKNN Model File
#     rknn.export_rknn('./digital_gesture.rknn')
#
#     # Release RKNN Context
#     rknn.release()
#
# #
#
rknn = RKNN(verbose=True)
INPUT_SIZE = 224
# Pre-process config
print('--> Config model')
rknn.config(mean_values=[0.0, 0.0, 0.0], std_values=[255.0, 255.0, 255.0], target_platform='rk3588')
print('done')

# Load model
print('--> Loading model')
ret = rknn.load_tensorflow(tf_pb='model/tf_model.pb',
                           # inputs=['vgg16_netvlad_pca/conv1_1/kernel'],
                           # outputs=['vgg16_netvlad_pca/l2_normalize_1'],
                           inputs=['vgg16_netvlad_pca/conv1_1/kernel'],
                           outputs=['vgg16_netvlad_pca/WPCA/bias'],

                           input_size_list=[[3,3,3,64]])
                           # input_size_list=[[1, INPUT_SIZE, INPUT_SIZE, 3]])
if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')

# Build Model
print('--> Building model')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')
#
# #
