from flask import Flask, render_template, Response, request
import cv2
import time
import numpy as np
import tensorflow as tf
from PIL import Image

global vid

app = Flask(__name__, template_folder='./templates')
 

def gen_frames():  # generate frame by frame from camera
        global vid, flag
    
        vid = cv2.VideoCapture(0)
        while True:
         ## read the camera frame
            success,frame=vid.read()
            if not success:
                break
            else:
                ret,buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()

            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def mask_detect(img_arr):
    new_model=tf.keras.models.load_model('my_model.h5')
    img=Image.fromarray(img_arr[2])
    #img.show()
    final_image= img.resize((224, 224))
    final_image=np.expand_dims(final_image,axis=0)
    final_image=final_image/255.0
    predictions= new_model.predict(final_image)
    return predictions[0][0]



@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response( gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests',methods=['POST','GET'])
def tasks():
    global  vid
    img_arr = []
    if request.method == 'POST':
        vid = cv2.VideoCapture(0)
        timeout = time.time() + 1
        while True:
            success, image = vid.read()
            if(time.time()>timeout):
                break
            if (success):
                img_arr.append(image)
            else:
                break
        vid.release()
        fail = mask_detect(img_arr) 
        print(fail)
        if( fail >= 0.9998 ):
            return render_template('index.html', result = "Please Wear a MASK!")
        else:
            return render_template('index.html', result = "Great! You are wearing a MASK")


if __name__ == '__main__':
    app.run()
