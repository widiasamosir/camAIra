from flask import Flask, request, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename


# Import Packages from Learning to See in the Dark

from __future__ import division
from matplotlib import image
from matplotlib import pyplot
import os, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import rawpy
import glob

app = Flask(__name__)

input_dir = './static/RAW/'
checkpoint_dir = './checkpoint/Sony/'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

#------------------------------------------------------------#fitur1#----------------------------------------------------------   

# upload selected image and forward to processing page
@app.route('/fiturpertama', methods = ['GET', 'POST'])
def upload_pertama():
    # create image directory if not found
    target = os.path.join(APP_ROOT, 'static/RAW/')
    if not os.path.isdir(target):
        os.mkdir(target)
   # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename


    # save file
    destination = "/".join([target, filename])
    print("File saved to to:", destination)
    upload.save(destination)
    # forward to processing page
    return render_template("processing.html", image_name=filename)



# retrieve file from directory
@app.route('/static/RAW/<filename>')
def send_image1(filename):
    return send_from_directory("static/RAW", filename)

#------------------------------------------------------------Learning to See in the Dark(Ltsitd)--------------------------------------
#------------------------------------------------------------Learning to See in the Dark(Ltsitd)--------------------------------------

@app.route('/predict_Ltsitd/<filename>', methods = ['GET', 'POST'])
def predict_Ltsitd (filename) :
    if request.method == 'POST':
        file_path = os.path.join(APP_ROOT, 'static/hasilfitur1')
        if not os.path.isdir(file_path):
            os.mkdir(file_path)

            DEBUG = 0
            if DEBUG == 1:
                save_freq = 2
                test_ids = test_ids[0:5]

            def lrelu(x):
                return tf.maximum(x * 0.2, x)


            def upsample_and_concat(x1, x2, output_channels, in_channels):
                pool_size = 2
                deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
                deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

                deconv_output = tf.concat([deconv, x2], 3)
                deconv_output.set_shape([None, None, None, output_channels * 2])

                return deconv_output


            def network(input):
                conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
                conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
                pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

                conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
                conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
                pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

                conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
                conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
                pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

                conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
                conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
                pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

                conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
                conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

                up6 = upsample_and_concat(conv5, conv4, 256, 512)
                conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
                conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

                up7 = upsample_and_concat(conv6, conv3, 128, 256)
                conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
                conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

                up8 = upsample_and_concat(conv7, conv2, 64, 128)
                conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
                conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

                up9 = upsample_and_concat(conv8, conv1, 32, 64)
                conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
                conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

                conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
                out = tf.depth_to_space(conv10, 2)
                return out


            def pack_raw(raw):
                # read image black level
                bl = raw.black_level_per_channel[0]

                # pack Bayer image to 4 channels
                im = raw.raw_image_visible.astype(np.float32)
                im = np.maximum(im - bl, 0) / (16383 - bl)  # subtract the black level

                im = np.expand_dims(im, axis=2)
                img_shape = im.shape
                H = img_shape[0]
                W = img_shape[1]

                out = np.concatenate((im[0:H:2, 0:W:2, :],
                                      im[0:H:2, 1:W:2, :],
                                      im[1:H:2, 1:W:2, :],
                                      im[1:H:2, 0:W:2, :]), axis=2)
                return out


            #manggil model
            sess = tf.Session()
            in_image = tf.placeholder(tf.float32, [None, None, None, 4])
            gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
            out_image = network(in_image)

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt:
                print('loaded ' + ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

            #proses gambar input
            in_files = glob.glob(input_dir + '/*.ARW')
            in_path = in_files[0]
            in_fn = os.path.basename(in_path)
            print(in_fn)

            ratio = 200
            raw = rawpy.imread(in_path)

            input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)

            # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
            scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
                    
            input_full = np.minimum(input_full, 1.0)

            output = sess.run(out_image, feed_dict={in_image: input_full})
            output = np.minimum(np.maximum(output, 0), 1)

            output = output[0, :, :, :]
            origin_full = scale_full[0, :, :, :]

    filename= origin_full, output

    return render_template("resultem.html", filename) #belom tau bisa gini apa engga wkwkw


# retrieve file from 'static/hasilfitur1' directory
@app.route('/static/hasilfitur1/<filename>')
def send_hasilfitur1(filename):
    return send_from_directory("static/hasilfitur1", filename)

#download images
@app.route('/static/hasilfitur1/<filename>',methods = ['GET'])
def unduh (filename):
    return send_file("static/hasilfitur1", filename, as_attachment=True)



# display gambar input
#pyplot.imshow(origin_full)
#pyplot.show()
# display gambar output
#pyplot.imshow(output)
#pyplot.show()

#-------------------------------------------------fitur pelengkap--------------------------------------------------------
# about
@app.route('/about')
def about():
     return render_template("tentang.html")


# help (MASUKIN DATABASE)

@app.route('/help')
def help():
    return render_template('help.html')

if __name__ == '__main__':
   app.run(debug = True)






