from __future__ import division
from flask import Flask, request, render_template, send_from_directory
from flask_ngrok import run_with_ngrok
import os
from werkzeug.utils import secure_filename



# Import Packages from ISR
import numpy as np
from PIL import Image
from ISR.models import RDN, RRDN


# Import Packages from Learning to See in the Dark


from matplotlib import image
from matplotlib import pyplot
import os, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import rawpy
import glob


app = Flask(__name__)
run_with_ngrok(app)   #starts ngrok when the app is run
#---------------------folder2 fitur1----------------------------------------------------------------------------------------------
input_dir = './static/RAW/'
checkpoint_dir = './checkpoint/Sony/'

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

#----------------------TAMPILAN AWAL WEB--------------------------------------------------------------------------------------------
@app.route('/')
def home():
   return render_template('base.html')

#------------------------------------------------------------#fitur1#----------------------------------------------------------   

# upload selected image and forward to processing page
@app.route('/fiturpertama', methods = ['GET', 'POST'])
def upload_pertama():
    # create image directory if not found
    target = os.path.join(APP_ROOT, 'static/RAW/')
    if not os.path.isdir(target):
        os.mkdir(target)
    targetdua = os.path.join(APP_ROOT, 'static/PNG/')
   # create image directory if not found
    if not os.path.isdir(targetdua):
        os.mkdir(targetdua)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename
    fileinput = os.path.splitext(filename)[0] + '.png'

    # save file
    destination = "/".join([target, filename])
    print("File saved to to:", destination)
    upload.save(destination)

    # save file png
    #destination = "/".join([targetdua, filename])
    #print("File saved to to:", destination)
    #upload.save(destination)

    input_dir = './static/RAW/'
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

    #proses gambar input
    in_files = glob.glob(input_dir + filename)
    in_path = in_files[0]

    raw = rawpy.imread(in_path)
    im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
    origin_full = scale_full[0, :, :, :]

    fileinput = os.path.splitext(filename)[0] + '.png'
    namainput=fileinput
    scipy.misc.toimage(origin_full * 255, high=255, low=0, cmin=0, cmax=255).save('./static/PNG/' + fileinput)

    # forward to processing page
    return render_template("processing1.html", image_name=filename, namainput=fileinput)



# retrieve file from directory
@app.route('/static/PNG/<fileinput>')
def send_image1(fileinput):
    return send_from_directory("static/PNG", fileinput)


#------------------------------------------------------------Learning to See in the Dark(Ltsitd)--------------------------------------
#------------------------------------------------------------Learning to See in the Dark(Ltsitd)--------------------------------------

@app.route('/predict_Ltsitd/<filename>', methods = ['GET', 'POST'])
def predict_Ltsitd (filename) :
    
    if request.method == 'POST':
        file_path1 = os.path.join(APP_ROOT, 'static/hasilfitur1')
        if not os.path.isdir(file_path1):
            os.mkdir(file_path1)

        input_dir = './static/RAW/'
        checkpoint_dir = './checkpoint/Sony/'
	

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
        in_files = glob.glob(input_dir + filename)
        in_path = in_files[0]
            
        #in_fn = os.path.basename(in_path)
        #print(in_fn)

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

        #save image
        fileoutput = os.path.splitext(filename)[0] + '.png'
		#fileoutput = namafile + '.png'
        fileinput = os.path.splitext(filename)[0] + '.png'
        namainput = fileinput	
        scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(file_path1 + '/' + fileoutput)
        #scipy.misc.toimage(origin_full * 255, high=255, low=0, cmin=0, cmax=255).save(input_dir + '/' + fileoutput)
    	#out = scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255)
        # out.save(os.path.join(file_path1, 'out.png'))
	
    keluar =  Image.open(os.path.join(file_path1, fileoutput))
	
    return render_template("resultem.html", keluar=fileoutput, namainput=fileinput) #belom tau bisa gini apa engga wkwkw

# display gambar input
#pyplot.imshow(origin_full)
#pyplot.show()
# display gambar output
#pyplot.imshow(output)
#pyplot.show()



# retrieve file from 'static/hasilfitur1' directory
@app.route('/static/hasilfitur1/<filename>')
def send_hasilfitur1(filename):
    fileoutput = os.path.splitext(filename)[0] + '.png'
    return send_from_directory("static/hasilfitur1", fileoutput)

#download images
@app.route('/static/hasilfitur1/<filename>',methods = ['GET'])
def unduh (filename):
    fileoutput = os.path.splitext(filename)[0] + '.png'
    return send_file("static/hasilfitur1", fileoutput, as_attachment=True)



# display gambar input
#pyplot.imshow(origin_full)
#pyplot.show()
# display gambar output
#pyplot.imshow(output)
#pyplot.show()



#------------------------------------------------------------#fitur2#--------------------------------------   

@app.route('/fiturkedua', methods = ['GET', 'POST'])
def upload_kedua():
    # create image directory if not found
    target = os.path.join(APP_ROOT, 'static/images/')
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
    return render_template("processing2.html", image_name=filename)


# retrieve file from directory
@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)



#------------------------------------------------------------predict ISR---------------------------------------------------
#------------------------------------------------------------predict ISR--------------------------------------------------
@app.route('/predict_sr/<filename>', methods = ['GET', 'POST'])
def predict_sr (filename) :
    if request.method == 'POST':
        file_path = os.path.join(APP_ROOT, 'static/output')
        if not os.path.isdir(file_path):
            os.mkdir(file_path)
        img = Image.open(os.path.join('static/images/', filename))
        #resize LR
        img.resize(size=(img.size[0]*4, img.size[1]*4), resample=Image.BICUBIC)
        #prediction
        
        lr_img = np.array(img)
        rrdn = RRDN(arch_params={'C': 4, 'D': 3, 'G': 32, 'G0': 32, 'x': 4, 'T': 10})
        rrdn.model.load_weights('weights/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5')
        sr_img = rrdn.predict(lr_img)
        Image.fromarray(sr_img)
        #save image
        sr = Image.fromarray(sr_img)
       
        sr.save(os.path.join(file_path, filename))
    return render_template("result.html",prediction=filename)

# retrieve file from 'static/output' directory
@app.route('/static/output/<filename>')
def send_output(filename):
    return send_from_directory("static/output", filename)

#download images
@app.route('/static/output/<filename>',methods = ['GET'])
def download (filename):
    return send_file("static/output", filename, as_attachment=True)


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
   app.run()
