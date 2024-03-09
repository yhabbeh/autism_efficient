from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
from keras.models import load_model
from skimage import transform
import cv2



app = Flask(__name__)
model = load_model('inception_model.h5')

@app.route('/')
def helloworld():
    return 'Prediction Home Page'

@app.route('/uploads', methods=['POST', 'GET'])
def upload_files():
    if request.method == 'POST':
        if 'images[]' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        files = request.files.getlist('images[]')
        result = multi_images_predict(files)
        if result >= .5:
            print("result", "True", "precentage", result)
            return jsonify({"response": "the user has an autism", "result": True, "precentage": result}), 200
        else:
            print("result", "False", "precentage", result)
            return jsonify({"response": "the user hasn't an autism", "result": False, "precentage": result}), 200
    else:
        return jsonify({"Error 405 ": "The method is not allowed for the requested URL this API is \'POST\'"}), 405


def multi_images_predict(files):
    counter = 0
    count_detect = 0
    for image in files:
        print('counter =>', counter)
        if prediction(face_crop_image(image=Image.open(BytesIO(image.read()))), model_autism=model):
           count_detect += 1
        counter += 1
    return count_detect / counter


def face_crop_image(image):
    image = np.array(image)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(image, minNeighbors=3)
    print(face)
    if len(face) != 0:
        for x, y, w, h in face:
            face = image[y:y + h, x:x + w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face
    return image

def prediction(image, model_autism):
    np_image = np.array(image).astype('float32') / 255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    img = tf.image.resize(np_image, (224, 224))  # Resize the image

    res = model_autism.predict(img)[0].argmax()
    return res == 1
