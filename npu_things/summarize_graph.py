import argparse

import tensorflow as tf

print('tf.__version__', tf.__version__)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# https://www.tensorflow.org/guide/extend/model_files#nodes
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#inspecting-graphs

# curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" | tar -xz

# bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=inception_v3_2016_08_28_frozen.pb
# Found 1 possible inputs: (name=input, type=float(1), shape=[1,299,299,3])
# No variables spotted.
# Found 1 possible outputs: (name=InceptionV3/Predictions/Reshape_1, op=Reshape)
# Found 23853946 (23.85M) const parameters, 0 (0) variable parameters, and 0 control_edges
# Op types used: 489 Const, 379 Identity, 188 Mul, 188 Add, 95 Conv2D, 94 Sub, 94 Rsqrt, 94 Relu, 15 ConcatV2, 10 AvgPool, 4 MaxPool, 2 Reshape, 1 BiasAdd, 1 Softmax, 1 Squeeze, 1 Placeholder

# TODO: add input/output shapes

def print_inputs(pb_filepath):
    with tf.gfile.GFile(pb_filepath, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        input_list = []
        for op in graph.get_operations(): # tensorflow.python.framework.ops.Operation
            if op.type == "Placeholder":
                input_list.append(op.name)

        print('Inputs:', input_list)


def print_outputs(pb_filepath):
    with open(pb_filepath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        name_list = []
        input_list = []
        for node in graph_def.node: # tensorflow.core.framework.node_def_pb2.NodeDef
            name_list.append(node.name)
            input_list.extend(node.input)

        outputs = set(name_list) - set(input_list)
        print('Outputs:', list(outputs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument()
    args = parser.parse_args()

    print_inputs(args.pb_filepath)
    print_outputs(args.pb_filepath)
