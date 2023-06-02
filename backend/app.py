from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import numpy as np
from werkzeug.exceptions import BadRequest

app = Flask(__name__)
CORS(app, origins='*')
model = tf.keras.models.load_model('./model/car-brand-model.h5')

# Classes for our Model
classes = {
    0: 'Hyundai',
    1: 'Lexus',
    2: 'Mazda',
    3: 'Mercedes',
    4: 'Opel',
    5: 'Skoda',
    6: 'Toyota',
    7: 'Volkswagen',
}


@app.route('/classify-image', methods=['POST'])
def classify_image():
    try:
        image = request.files['image']
        image_pil = Image.open(image)
        image_resized = image_pil.resize((240, 196))
        image_gray = image_resized.convert('L')
        image_array = np.array(image_gray)
        image_array_rgb = np.repeat(image_array[:, :, np.newaxis], 3, axis=-1)

        prediction = model.predict(np.expand_dims(image_array_rgb, axis=0))
        predicted_class = np.argmax(prediction)

        result = {
            'class': int(predicted_class),
            'class_name': classes.get(predicted_class),
        }

        return jsonify(result)
    except BadRequest as e:
        error_message = str(e)
        return jsonify({'error': error_message}), 400

if __name__ == '__main__':
    app.run()
