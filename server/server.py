# %%
from flask import Flask, request, jsonify
import FaceDetection as FD

app = Flask(__name__)


@app.route('/detect', methods=['GET', 'POST', 'OPTIONS'])
def detect():
    image_data = request.form['image_data']

    response = jsonify(FD.face(image_data))

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    print("Face detections server started")
    app.run(port=5502)


# %%
