import cv2
from flask import Flask, request
from flask_cors import CORS
import numpy as np
from io import BytesIO
import tensorflow as tf
from PIL import Image
from skimage import transform
from keras.models import load_model
from flask import request, jsonify

app = Flask(__name__)
CORS(app)


@app.route('/')
def helloworld():
    return 'Prediction Home Page'


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.getlist('images[]')
        print(file)
        # for imagesingle in f:
        #     image = Image.open(BytesIO(imagesingle.read()))
        #     print(predict(face_crop_image(image), model))
        # if predict(face_crop_image(image), model):
        #     return {"response": "the user has an autism", "result": "True"}
        # else:
        #     return {"response": "the user hasn't an autism", "result": "False"}
        # return send_from_direcaory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


def multi_images_predict(filenames):
    counter = 0
    count_detect = 0
    for i in filenames:
        print(counter)
        if prediction(face_crop_image(Image.open(BytesIO(i.read()))), model_autism=model):
            count_detect += 1
        counter += 1
    return count_detect / counter


@app.route('/uploads', methods=['POST', 'GET'])
def upload_files():
    if request.method == 'POST':
        if 'images[]' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        files = request.files.getlist('images[]')
        result = multi_images_predict(files)
        if result >= .5:
            return {"response": "the user has an autism", "result": "True", "precentage": result}
        else:
            return {"response": "the user hasn't an autism", "result": "False"}
    else:
        return jsonify({"Error 405 ": "The method is not allowed for the requested URL this API is \'POST\'"}), 405


def prediction(image, model_autism):
    np_image = np.array(image).astype('float32') / 255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    img = tf.image.resize(np_image, (224, 224))  # Resize the image
    res = model_autism.predict(img)[0].argmax()
    print(res)
    return res == 1


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


if __name__ == '__main__':
    model = load_model('inception_model.h5')
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(host='0.0.0.0', port=5000)
