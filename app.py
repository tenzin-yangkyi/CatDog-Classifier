from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array  # type: ignore
from keras.utils import load_img
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = tf.keras.models.load_model(
    "model/cats_dogs_cnn_model.keras"
)

def predict_image(img_path):

    img = load_img(img_path, target_size=(128,128))

    img_array = img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        label = "Dog"
        confidence = confidence * 100
    else:
        label = "Cat"
        confidence = (1-confidence) * 100

    return label, round(confidence,2)

@app.route("/", methods=["GET","POST"])
def home():

    result = None
    confidence = None
    image_file = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(path)

            result, confidence = predict_image(path)

            image_file = path

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_file=image_file
    )

if __name__ == "__main__":
    app.run(debug=True)