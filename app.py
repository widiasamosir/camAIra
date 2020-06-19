from flask import Flask, request, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
# Import Packages from ISR
import numpy as np
from PIL import Image
from ISR.models import RDN, RRDN
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "collect/file"
app.config['MAX_CONTENT_PATH'] = "1000000000 bytes"
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def home():
   return render_template('base.html')

#------------------------------------------------------------#fitur1#--------------------------------------   

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

#-----------------------------------------------------------------------------


#------------------------------------------------------------predict ISR-------------------------------------- 
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
#dowload images
@app.route('/static/output/<filename>',methods = ['GET', 'POST'])
def download (filename):
    return send_file("static/output", filename, as_attachment=True)

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










#(kodingan buat masukin ke email.. tapi masih bingung gimanahh)
    #Fullname = request.form.get("full_name")
    #email = request.form.get("email")
    #message =request.form.get("message")

    #message = "sent to ellena, amanda, widia!"
    #server= smtplib.SMTP("smtp.gmail.com", 587)
    #server.starttls()
