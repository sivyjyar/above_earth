import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import csv

import netvlad_tf.net_from_mat as nfm
import netvlad_tf.nets as nets

tf.disable_v2_behavior()

tf.reset_default_graph()

image_batch = tf.placeholder(
        dtype=tf.float32, shape=[None, None, None, 3])

net_out = nets.vgg16NetvladPca(image_batch)
saver = tf.train.Saver() #создает место для хранения переменных


sess = tf.Session() # класс для работы операций из тензорфлоу
saver.restore(sess, nets.defaultCheckpoint())  #восстановление весов системы


#############################################################



def similarity(vector1, vector2):
    return np.dot(vector1, vector2.T) / np.dot(np.linalg.norm(vector1, axis=1, keepdims=True),
                                               np.linalg.norm(vector2.T, axis=0, keepdims=True))

values = range(1,9,1)
val_parts = range(0, 5, 1)


inim = cv2.imread('images/mgu_w.jpg')
inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)


batch = np.expand_dims(inim, axis=0)
result = sess.run(net_out, feed_dict={image_batch: batch}) # расчет графов

print(result)

# with open('res.csv', 'w') as f:
#     writer = csv.writer(f)
#     header = ['point', '1-2', '1-3', '1-4', '1-5']
#     writer.writerow(header)
#
#     for i in values:
#         f_data = [i]
#         for j in val_parts:
#             name = 'parts' + str(j) + '/' + str(2 * i) + '0_part.jpg'
#             inim = cv2.imread(name)
#             inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
#             batch = np.expand_dims(inim, axis=0)
#             result = sess.run(net_out, feed_dict={image_batch: batch})
#
#             if j == 0:
#                 a = result
#             else:
#                 f_data.append(float(similarity(a, result)[0]))
#         writer.writerow(f_data)

        # name = 'parts/' + str(2 * i) + '0_part.jpg'
        # inim = cv2.imread(name)
        # inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
        #
        # batch = np.expand_dims(inim, axis=0)
        # result = sess.run(net_out, feed_dict={image_batch: batch})
        #
        # a = result
        #
        # name = 'parts1/' + str(2 * i) + '0_part.jpg'
        # inim = cv2.imread(name)
        # inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
        #
        # batch = np.expand_dims(inim, axis=0)
        # result = sess.run(net_out, feed_dict={image_batch: batch})
        #
        # a1 = result
        #
        # name = 'parts2/' + str(2 * i) + '0_part.jpg'
        # inim = cv2.imread(name)
        # inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
        #
        # batch = np.expand_dims(inim, axis=0)
        # result = sess.run(net_out, feed_dict={image_batch: batch})
        #
        # a2 = result
        # f_data = [i, float(similarity(a, a1)[0]), float(similarity(a, a2)[0])]
        # writer.writerow(f_data)


# inim = cv2.imread('mgu_w.jpg')
# inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
#
#
# batch = np.expand_dims(inim, axis=0)
# result = sess.run(net_out, feed_dict={image_batch: batch}) # расчет графов
#
# a1 = result
#
# inim = cv2.imread('usa.jpg')
# inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
#
#
# batch = np.expand_dims(inim, axis=0)
# result = sess.run(net_out, feed_dict={image_batch: batch}) # расчет графов
#
# a2 = result

# inim = cv2.imread(nfm.exampleImgPath())
# inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
#
#
# batch = np.expand_dims(inim, axis=0)
# result = sess.run(net_out, feed_dict={image_batch: batch}) # расчет графов


