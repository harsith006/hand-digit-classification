import streamlit as st
import numpy as np
import os
import pickle
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split
from streamlit_drawable_canvas import st_canvas

# ----------- Classifier Logic -----------
class DigitClassifier:
    def __init__(self):
        self.model = None
        self.trained = False
        self.data_loaded = False

    def build_model(self):
        self.model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)

    def load_data(self):
        try:
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
            X = mnist.data.astype('float32') / 255.0
            y = mnist.target.astype('int')
        except:
            digits = load_digits()
            X = digits.data / 16.0
            y = digits.target

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.data_loaded = True

    def train(self):
        if not self.data_loaded:
            self.load_data()
        if self.model is None:
            self.build_model()

        self.model.fit(self.x_train, self.y_train)
        self.trained = True
        return self.model.score(self.x_test, self.y_test)

    def predict(self, image_array):
        if not self.trained:
            return None
        return self.model.predict(image_array.reshape(1, -1))[0]

    def save(self, filepath="digit_rf_model.pkl"):
        if self.trained:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)

    def load(self, filepath="digit_rf_model.pkl"):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            self.trained = True
            return True
        return False

# ----------- Streamlit UI -----------
st.set_page_config(page_title="Digit Recognizer", layout="wide")
st.title("🧠 Handwritten Digit Recognition")

# Initialize classifier
if "classifier" not in st.session_state:
    st.session_state.classifier = DigitClassifier()

classifier = st.session_state.classifier

# Sidebar controls
with st.sidebar:
    st.header("Model Controls")

    if st.button("🧪 Train Model"):
        accuracy = classifier.train()
        classifier.save()
        st.success(f"Model trained with accuracy: {accuracy*100:.2f}%")

    if st.button("📂 Load Saved Model"):
        if classifier.load():
            st.success("Model loaded successfully.")
        else:
            st.warning("No saved model found.")

# Drawing canvas
st.subheader("✏️ Draw a digit below (0–9)")
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess(image_data):
    image = Image.fromarray(image_data).convert("L")
    image = image.resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    return img_array

if st.button("🔍 Predict Digit"):
    if canvas_result.image_data is not None:
        img_array = preprocess(canvas_result.image_data)
        if classifier.trained:
            pred = classifier.predict(img_array)
            st.success(f"Predicted Digit: **{pred}**")
        else:
            st.error("Model not trained or loaded.")
    else:
        st.warning("Draw something to predict.")
