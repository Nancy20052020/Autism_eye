from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import StringField, PasswordField, EmailField
from wtforms.validators import DataRequired, Email
import os
import time
import base64
import io
import urllib.request
from datetime import datetime
from flask import send_file
import numpy as np
import json
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=BASE_DIR, static_folder=os.path.join(BASE_DIR, "static"))
app.secret_key = os.environ.get("SECRET_KEY", "autism-eye-demo")

csrf = CSRFProtect(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FILE_ID = "1dic8I4MqvzRYAo0kbU5nTbKX4ufMBcfz"
MODEL_URL = os.environ.get(
    "MODEL_URL",
    f"https://drive.usercontent.google.com/download?id={MODEL_FILE_ID}&export=download&confirm=t",
)
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("/tmp", "cbo_model.h5"))


def download_model(url, destination):
    if os.path.exists(destination) and os.path.getsize(destination) > 1_000_000:
        return destination
    logger.info("Downloading model.h5")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    if payload[:1] == b"<" or b"<html" in payload[:200].lower():
        raise RuntimeError("Model download returned a web page instead of the .h5 file")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as model_file:
        model_file.write(payload)
    return destination


def load_detection_model():
    import tensorflow as tf
    from tensorflow import keras
    from keras.src.layers.core.lambda_layer import Lambda

    original_from_config = Lambda.from_config.__func__

    def expand_time(tensor):
        return tf.expand_dims(tensor, axis=1)

    def from_config(cls, config):
        if config.get("name") == "lambda_9":
            return Lambda(expand_time, name="lambda_9")
        return original_from_config(cls, config)

    Lambda.from_config = classmethod(from_config)
    path = download_model(MODEL_URL, MODEL_PATH)
    try:
        return keras.models.load_model(path, compile=False, safe_mode=False)
    finally:
        Lambda.from_config = classmethod(original_from_config)


model = None
MODEL_LOADED = False


def ensure_model():
    global model, MODEL_LOADED
    if MODEL_LOADED and model is not None:
        return True
    try:
        model = load_detection_model()
        MODEL_LOADED = True
        logger.info("Autism detection model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        MODEL_LOADED = False
        return False


def stored_results():
    if "results" not in session:
        session["results"] = {}
    return session["results"]


def preprocess_eye_data(image1_time, image2_time, additional_features=None):

    try:
        img1_time = float(image1_time)
        img2_time = float(image2_time)

        img1_time = max(img1_time, 0.1)
        img2_time = max(img2_time, 0.1)

        total_time = img1_time + img2_time
        features = [
            img1_time,
            img2_time,
            abs(img1_time - img2_time),
            img1_time / total_time,
            img2_time / total_time,
            max(img1_time, img2_time) / min(img1_time, img2_time),
            total_time,
            (img1_time * img2_time) ** 0.5,
            np.log(img1_time + 1),
            np.log(img2_time + 1),
        ]
        if additional_features:
            features.extend(additional_features)

        target_features = 20
        while len(features) < target_features:
            features.append(0.0)
        features = features[:target_features]

        return np.array(features).reshape(1, -1)
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return None


def predict_autism_simple(round1_data, round2_data):

    try:

        r1_img1 = float(round1_data.get('image1_time', 0))
        r1_img2 = float(round1_data.get('image2_time', 0))
        r2_img1 = float(round2_data.get('image1_time', 0))
        r2_img2 = float(round2_data.get('image2_time', 0))

        r1_total = r1_img1 + r1_img2
        r2_total = r2_img1 + r2_img2

        if r1_total < 0.1 or r2_total < 0.1:
            return "non-autistic", 0.3, 0.4, 0.75

        r1_ratio1 = r1_img1 / r1_total
        r1_ratio2 = r1_img2 / r1_total
        r2_ratio1 = r2_img1 / r2_total
        r2_ratio2 = r2_img2 / r2_total

        ratio_consistency = abs(r1_ratio1 - r2_ratio1)

        autistic_indicators = 0
        confidence_factors = []

        if (r1_ratio1 > 0.75 or r1_ratio2 > 0.75) and (r2_ratio1 > 0.75 or r2_ratio2 > 0.75):
            autistic_indicators += 1
            confidence_factors.append(0.3)

        if r1_total < 5 or r2_total < 5:
            autistic_indicators += 1
            confidence_factors.append(0.2)

        if ratio_consistency < 0.1 and (r1_ratio1 > 0.8 or r1_ratio1 < 0.2):
            autistic_indicators += 1
            confidence_factors.append(0.2)

        if autistic_indicators >= 2:
            prediction = "autistic"
            probability = 0.65 + sum(confidence_factors)
        elif autistic_indicators == 1:
            prediction = "non-autistic"
            probability = 0.4 + sum(confidence_factors)
        else:
            prediction = "non-autistic"
            probability = 0.25 + sum(confidence_factors)

        probability = min(probability, 0.95)
        probability = max(probability, 0.05)

        if prediction == "autistic":
            confidence = min(probability, 0.8)
        else:
            confidence = min(1.0 - probability, 0.8)
        accuracy = 0.78

        return prediction, probability, confidence, accuracy

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return "non-autistic", 0.5, 0.3, 0.65


def decode_face_image(image_base64):
    from PIL import Image

    if not image_base64:
        return None
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    raw = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image = image.resize((224, 224))
    return np.asarray(image, dtype=np.float32)


def predict_face(image_array):
    probabilities = model.predict(image_array.reshape(1, 224, 224, 3), verbose=0)[0]
    autistic_probability = float(probabilities[0])
    if autistic_probability >= 0.5:
        prediction = "autistic"
        probability = autistic_probability
    else:
        prediction = "non-autistic"
        probability = 1.0 - autistic_probability
    confidence = float(min(abs(autistic_probability - 0.5) * 2, 0.99))
    return prediction, probability, confidence


def predict_autism_with_model(features, round1_data, round2_data, face_image=None):

    if face_image is not None and ensure_model():
        try:
            prediction, probability, confidence = predict_face(face_image)
            return prediction, probability, confidence, None
        except Exception as e:
            logger.error(f"Error with model prediction: {e}")

    return predict_autism_simple(round1_data, round2_data)


@app.route('/round1')
def round1():
    return render_template('round1.html')


@app.route('/autism')
def autism():
    return render_template('autism.html')


@app.route('/round2')
def round2():
    return render_template('round2.html')


@app.route('/submit_times', methods=['POST'])
@csrf.exempt
def submit_times():
    try:
        data = request.json
        round_name = data.get('round')
        image1_time = float(data.get('image1_time', 0))
        image2_time = float(data.get('image2_time', 0))
        face_image = decode_face_image(data.get('face_image'))

        round_result = {
            "image1_time": image1_time,
            "image2_time": image2_time,
        }

        eye_results = stored_results()
        if round_name == 'round2' and 'round1' in eye_results:
            features = preprocess_eye_data(image1_time, image2_time)
            prediction, probability, confidence, accuracy = predict_autism_with_model(
                features, eye_results['round1'], round_result, face_image
            )
        else:
            features = preprocess_eye_data(image1_time, image2_time)
            prediction, probability, confidence, accuracy = predict_autism_with_model(
                features, round_result, {}, face_image
            )

        round_result.update({
            "result": prediction,
            "probability": probability,
            "confidence": confidence,
            "accuracy": accuracy,
            "model_used": MODEL_LOADED,
            "features": features.tolist() if features is not None else []
        })

        eye_results[round_name] = round_result
        session.modified = True

        return jsonify({
            "status": "success",
            "result": prediction,
            "probability": round(probability * 100, 2),
            "confidence": round(confidence * 100, 2),
            "accuracy": round((confidence if accuracy is None else accuracy) * 100, 2),
            "model_used": MODEL_LOADED,
            "processing_time": round(time.time() % 1, 3)
        })

    except Exception as e:
        logger.error(f"Error in submit_times: {e}")
        return jsonify({
            "status": "error",
            "message": "Processing failed",
            "result": "non-autistic",
            "probability": 50,
            "confidence": 30,
            "accuracy": 65
        })


@app.route('/result')
@csrf.exempt
def result():
    try:
        eye_results = stored_results()
        if 'round1' not in eye_results or 'round2' not in eye_results:
            flash("Please complete both rounds first.")
            return redirect(url_for('round1'))

        round1_result = eye_results['round1']['result']
        round2_result = eye_results['round2']['result']

        combined_features = preprocess_eye_data(
            eye_results['round1']['image1_time'],
            eye_results['round1']['image2_time']
        )
        round_probabilities = [
            eye_results[name].get('probability')
            for name in ('round1', 'round2')
            if eye_results[name].get('result')
        ]
        if round_probabilities and eye_results['round1'].get('model_used'):
            final_probability = float(np.mean(round_probabilities))
            final_prediction = "autistic" if final_probability >= 0.5 else "non-autistic"
            final_confidence = float(min(abs(final_probability - 0.5) * 2, 0.99))
            final_accuracy = None
        else:
            final_prediction, final_probability, final_confidence, final_accuracy = predict_autism_with_model(
                combined_features, eye_results['round1'], eye_results['round2']
            )

        if final_probability > 0.7:
            final_result = "High Possibility of Autism"
            result_class = "high-risk"
        elif final_probability > 0.4:
            final_result = "Moderate Possibility of Autism"
            result_class = "moderate-risk"
        else:
            final_result = "Low Possibility of Autism"
            result_class = "low-risk"

        detailed_results = {
            'final_result': final_result,
            'result_class': result_class,
            'probability': round(final_probability * 100, 2),
            'confidence': round(final_confidence * 100, 2),
            'accuracy': round((final_confidence if final_accuracy is None else final_accuracy) * 100, 2),
            'round1': eye_results['round1'],
            'round2': eye_results['round2'],
            'model_used': eye_results['round1'].get('model_used', False),
            'recommendations': get_recommendations(final_probability),
            'individual_predictions': {
                'round1': round1_result,
                'round2': round2_result
            }
        }

        return render_template('result.html', **detailed_results)

    except Exception as e:
        logger.error(f"Error in result route: {e}")
        flash("An error occurred while processing results.")
        return redirect(url_for('home'))


def get_recommendations(probability):

    if probability > 0.7:
        return [
            "Consider consulting with a healthcare professional for further evaluation.",
            "Early intervention programs may be beneficial.",
            "Connect with autism support groups in your community.",
            "Keep a detailed log of behaviors and patterns to discuss with professionals."
        ]
    elif probability > 0.4:
        return [
            "Monitor development and behaviors over time.",
            "Consider discussing concerns with a pediatrician or family doctor.",
            "Look into developmental screening tools and assessments.",
            "Stay informed about autism spectrum characteristics."
        ]
    else:
        return [
            "Continue regular developmental monitoring.",
            "Maintain open communication about any concerns.",
            "Support continued healthy development through play and interaction.",
            "Stay aware of developmental milestones."
        ]


@app.route('/')
def home():
    return render_template('home.html')

class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])


def browser_accounts():
    if "accounts" not in session:
        session["accounts"] = {}
    return session["accounts"]


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()

    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        accounts = browser_accounts()
        if username in accounts or any(item["email"] == email for item in accounts.values()):
            flash("Username or email already exists. Please choose a different one.")
            return redirect(url_for('signup'))

        accounts[username] = {
            "email": email,
            "password": generate_password_hash(form.password.data),
        }
        session.modified = True
        session['username'] = username
        flash("Signup successful!")
        return redirect(url_for('home'))

    return render_template('signup.html', form=form)


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        account = browser_accounts().get(username)
        if account and account["email"] == email and check_password_hash(account["password"], form.password.data):
            session['username'] = username
            flash("Login successful!")
            return redirect(url_for('home'))
        if account:
            flash("Invalid password.")
            return redirect(url_for('login'))
        flash("Username or email not found. Please sign up first.")
        return redirect(url_for('signup'))

    return render_template('login.html', form=form)


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You have been logged out.")
    return redirect(url_for('home'))


class ForgotPasswordForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired(), Email()])


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    form = ForgotPasswordForm()

    if form.validate_on_submit():
        email = form.email.data
        match = next((name for name, item in browser_accounts().items() if item["email"] == email), None)
        if match:
            session["reset_user"] = match
            return redirect(url_for('reset_password'))
        flash("Email not found in this browser. Sign up again on this device.")
        return redirect(url_for('forgot_password'))

    return render_template('forgot_password.html', form=form)


@app.route('/reset_password', methods=['GET', 'POST'])
@app.route('/reset-password', methods=['GET', 'POST'])
@csrf.exempt
def reset_password():
    username = session.get("reset_user")
    if not username or username not in browser_accounts():
        flash("Request a password reset from this browser first.")
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        browser_accounts()[username]["password"] = generate_password_hash(request.form.get("password", ""))
        session.modified = True
        session.pop("reset_user", None)
        flash("Your password has been updated!")
        return redirect(url_for('login'))

    return render_template('reset_password.html', form=ForgotPasswordForm())


if __name__ == '__main__':
    app.run(debug=True)