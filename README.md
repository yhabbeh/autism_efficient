**Autism Prediction API**

**Introduction**

This Flask application offers a preliminary approach to classifying images as potentially indicating autism using a pre-trained deep learning model. It accommodates uploading multiple images for analysis.

**Requirements**

* Python 3.11
* Flask
* TensorFlow
* Keras
* NumPy
* Pillow (PIL Fork)
* scikit-image
* OpenCV

**Installation**

1. Ensure you have Python and pip installed.
2. Create a virtual environment (recommended) to isolate project dependencies:
   ```bash
   python3 -m venv env
   source env/bin/activate  # Linux/macOS
   env\Scripts\activate.bat  # Windows
   ```
3. Install required libraries:
   ```bash
   pip install Flask tensorflow keras numpy Pillow scikit-image opencv-python
   ```

**Usage**

1. Clone this repository.
2. Navigate to the project directory.
3. Run the application:
   ```bash
   python main.py
   ```
   This starts the Flask development server, typically accessible at `http://127.0.0.1:5000/` (replace with your local IP if necessary).

4. Upload Images:
   - Use a tool like Postman or curl to send a POST request to `http://localhost:5000/uploads` with image files included under the key `'images[]'`.
   - Some API testing tools may allow uploading multiple files as well.
   - The response will be a JSON object with the predicted probability of autism for each image.

**Code Structure**

* `main.py`: Main Flask application file
* `inception_model.h5`: Pre-trained autism classification model
* `haarcascade_frontalface_default.xml`: Face detection cascade classifier

**API Endpoints**

* **`/` (GET):** Returns a simple welcome message.
* **`/uploads` (POST):**
    - Processes uploaded images.
    - Returns a JSON response with:
        - Prediction for each image (autism or not).
        - Confidence score (probability).
    - Supports uploading multiple images.

**Explanation**

- The application loads a pre-trained deep learning model from `inception_model.h5` for autism classification.
- It leverages OpenCV's face detection to potentially crop faces from images before prediction.
- The `multi_images_predict` function handles processing multiple images efficiently.
- Individual image prediction is handled by the `prediction` function, which resizes, normalizes, and potentially resizes (depending on implementation) the image before feeding it to the model.

**Disclaimer**

This is a basic example for experimentation and should not be used for medical diagnosis. Professional evaluation is necessary for any autism-related concerns.

This Readme provides a high-level overview of the project. Refer to the source code (`main.py`) for detailed implementation.

**Contributing**

We welcome pull requests for improvements, bug fixes, or enhancements to this project.
