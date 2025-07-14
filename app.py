from __future__ import division, print_function


import sys
import os
import glob
import re

import numpy as np


import tensorflow as tf
# Keras
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

path = "https://drive.google.com/file/d/1er_d9jY6kVu9tLv6Nxxl_A2zED0dJ4kU/view?usp=sharing"

# Model saved with Keras model.save()
MODEL_PATH = "Best_Model_IceptionV3.h5"


# Load your trained model

model = load_model("Deployment-of-YVMD-model/Best_Model_IceptionV3.h5")

# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    res = np.argmax(preds)
    return res


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        res = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        classes = ["Diseased Leaf","Fresh Leaf"]   # ImageNet Decode
        result = classes[res]           # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

