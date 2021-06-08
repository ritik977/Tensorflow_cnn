#Stage 1: Import all project dependencies

import os
import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from scipy.misc import imread, imsave
from flask import Flask, request, jsonify

print(tf.__version__)

#Stage 2: Load the pretrained model
with open('ritikagrawal_cnn_model.json', 'r') as f:
    model_json = f.read()
batch_size = 32
img_height = 340
img_width = 340

model = tf.keras.models.model_from_json(model_json)
# load weights into new model
model.load_weights("ritikagrawal_cnn_model.h5")

#Stage 3: Creating the Flask API
#Starting the Flask application
app = Flask(__name__)

#Defining the classify_image function
@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):
    #Define the uploads folder
    upload_dir = "test/test/"
    #Load an uploaded image
    image = imread(upload_dir + img_name)
    fruit_path = upload_dir + img_name
    #Define the list of class names 
    classes = ["apple", "banana", "mixed", "orange"]
# -------------------------------------------------------
    img = keras.preprocessing.image.load_img(fruit_path, target_size=(img_height, img_width))
    

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])



    #Perform predictions with pre-trained model
    #prediction = model.predict([image.reshape(1, i*500*3)])

    #Return the prediction to the user
    return jsonify({"object_identified":classes[np.argmax(predictions[0])]})

@app.route("/home", methods=["GET"])
def home():
    return jsonify("This is TDCX home page")

#Start the Flask application
app.run(port=5000, debug=False)

