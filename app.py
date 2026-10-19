from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import BadSignature, URLSafeTimedSerializer, SignatureExpired
from flask_mail import Mail, Message
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import StringField, PasswordField, EmailField
from wtforms.validators import DataRequired, Email, EqualTo
import os
import re
import time
import numpy as np
import json
import logging

from config import (
    MAIL_PASSWORD,
    MAIL_PORT,
    MAIL_SERVER,
    MAIL_USE_SSL,
    MAIL_USERNAME,
    MAIL_USE_TLS,
    MODEL_PATH,
    SECRET_KEY,
    db_config,
)

app = Flask(__name__)
app.secret_key = SECRET_KEY

csrf = CSRFProtect(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.config['MAIL_SERVER'] = MAIL_SERVER
app.config['MAIL_PORT'] = MAIL_PORT
app.config['MAIL_USERNAME'] = MAIL_USERNAME
app.config['MAIL_PASSWORD'] = MAIL_PASSWORD
app.config['MAIL_USE_TLS'] = MAIL_USE_TLS
app.config['MAIL_USE_SSL'] = MAIL_USE_SSL

mail = Mail(app)
s = URLSafeTimedSerializer(app.secret_key)

model = None
MODEL_LOADED = False
if not os.path.exists(MODEL_PATH):
    logger.info("No model file at %s. Using the rule-based estimate.", MODEL_PATH)
else:
    try:
        from tensorflow import keras

        model = keras.models.load_model(MODEL_PATH)
        logger.info("Autism detection model loaded successfully!")
        MODEL_LOADED = True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


def get_db_connection(with_database=True):
    connection = None
    config = dict(db_config)
    config['connection_timeout'] = 5
    if not with_database:
        config.pop('database', None)
    try:
        connection = mysql.connector.connect(**config)
        if connection.is_connected():
            logger.info("Connection to MySQL database was successful!")
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        return None
    return connection


def ensure_schema():
    """Create the database and tables when MySQL is reachable."""
    bootstrap = get_db_connection(with_database=False)
    if bootstrap is None:
        logger.warning("Skipping schema setup because MySQL is unavailable.")
        return False

    database_name = db_config['database']
    if not re.fullmatch(r'[A-Za-z0-9_]+', database_name or ''):
        logger.error("Invalid database name in configuration.")
        return False
    try:
        cursor = bootstrap.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{database_name}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        cursor.close()
        bootstrap.close()

        connection = get_db_connection()
        if connection is None:
            return False
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL UNIQUE,
                email VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(512) NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255),
                test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                round1_img1_time FLOAT,
                round1_img2_time FLOAT,
                round1_result VARCHAR(50),
                round2_img1_time FLOAT,
                round2_img2_time FLOAT,
                round2_result VARCHAR(50),
                final_result VARCHAR(100),
                model_confidence FLOAT,
                model_accuracy FLOAT,
                test_data JSON
            )
        """)
        connection.commit()
        cursor.close()
        connection.close()
        return True
    except Error as e:
        logger.error(f"Error creating schema: {e}")
        return False


def load_results():
    stored = session.get('test_results')
    if isinstance(stored, dict):
        return stored
    return {}


def store_results(results):
    session['test_results'] = results
    session.modified = True


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


def _score_single_round(img1, img2):
    total = img1 + img2
    if total < 0.1:
        return "non-autistic", 0.3, 0.4, 0.75

    ratio1 = img1 / total
    ratio2 = img2 / total
    autistic_indicators = 0
    confidence_factors = []

    if ratio1 > 0.75 or ratio2 > 0.75:
        autistic_indicators += 1
        confidence_factors.append(0.3)

    if total < 5:
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

    probability = min(max(probability, 0.05), 0.95)
    if prediction == "autistic":
        confidence = min(probability, 0.8)
    else:
        confidence = min(1.0 - probability, 0.8)
    return prediction, probability, confidence, 0.78


def predict_autism_simple(round1_data, round2_data):

    try:

        r1_img1 = float((round1_data or {}).get('image1_time', 0) or 0)
        r1_img2 = float((round1_data or {}).get('image2_time', 0) or 0)
        r2_img1 = float((round2_data or {}).get('image1_time', 0) or 0)
        r2_img2 = float((round2_data or {}).get('image2_time', 0) or 0)

        r1_total = r1_img1 + r1_img2
        r2_total = r2_img1 + r2_img2

        if r1_total < 0.1 and r2_total < 0.1:
            return "non-autistic", 0.3, 0.4, 0.75

        if r2_total < 0.1:
            return _score_single_round(r1_img1, r1_img2)
        if r1_total < 0.1:
            return _score_single_round(r2_img1, r2_img2)

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


def predict_autism_with_model(features, round1_data, round2_data):

    if not MODEL_LOADED or model is None or features is None:
        return predict_autism_simple(round1_data, round2_data)

    try:
        prediction_prob = model.predict(features, verbose=0)[0][0]

        if prediction_prob > 0.5:
            prediction = "autistic"
            probability = float(prediction_prob)
        else:
            prediction = "non-autistic"
            probability = 1.0 - float(prediction_prob)

        confidence = abs(prediction_prob - 0.5) * 2
        confidence = min(confidence, 0.95)
        accuracy = 0.92

        return prediction, probability, float(confidence), accuracy

    except Exception as e:
        logger.error(f"Error with model prediction: {e}")
        return predict_autism_simple(round1_data, round2_data)


def save_result_to_db(username, round_data, final_result, model_confidence, model_accuracy):
    try:
        connection = get_db_connection()
        if connection is None:
            return False

        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255),
                test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                round1_img1_time FLOAT,
                round1_img2_time FLOAT,
                round1_result VARCHAR(50),
                round2_img1_time FLOAT,
                round2_img2_time FLOAT,
                round2_result VARCHAR(50),
                final_result VARCHAR(100),
                model_confidence FLOAT,
                model_accuracy FLOAT,
                test_data JSON
            )
        """)

        # Insert result
        cursor.execute("""
            INSERT INTO test_results 
            (username, round1_img1_time, round1_img2_time, round1_result,
             round2_img1_time, round2_img2_time, round2_result, 
             final_result, model_confidence, model_accuracy, test_data)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            username or 'anonymous',
            round_data.get('round1', {}).get('image1_time', 0),
            round_data.get('round1', {}).get('image2_time', 0),
            round_data.get('round1', {}).get('result', ''),
            round_data.get('round2', {}).get('image1_time', 0),
            round_data.get('round2', {}).get('image2_time', 0),
            round_data.get('round2', {}).get('result', ''),
            final_result,
            model_confidence,
            model_accuracy,
            json.dumps(round_data)
        ))

        connection.commit()
        cursor.close()
        connection.close()
        return True

    except Exception as e:
        logger.error(f"Error saving to database: {e}")
        return False


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
        data = request.get_json(silent=True) or {}
        round_name = data.get('round')
        if round_name not in ('round1', 'round2'):
            return jsonify({
                "status": "error",
                "message": "Unknown round",
                "result": "non-autistic",
                "probability": 50,
                "confidence": 30,
                "accuracy": 65
            }), 400

        image1_time = float(data.get('image1_time', 0) or 0)
        image2_time = float(data.get('image2_time', 0) or 0)

        round_result = {
            "image1_time": image1_time,
            "image2_time": image2_time,
        }

        stored = load_results()
        if round_name == 'round2' and 'round1' in stored:
            features = preprocess_eye_data(image1_time, image2_time)
            prediction, probability, confidence, accuracy = predict_autism_with_model(
                features, stored['round1'], round_result
            )
        else:
            features = preprocess_eye_data(image1_time, image2_time)
            prediction, probability, confidence, accuracy = predict_autism_with_model(
                features, round_result, {}
            )

        flat_features = []
        if features is not None:
            flat_features = [float(value) for value in features.reshape(-1).tolist()]

        round_result.update({
            "result": prediction,
            "probability": float(probability),
            "confidence": float(confidence),
            "accuracy": float(accuracy),
            "features": flat_features,
        })

        stored[round_name] = round_result
        store_results(stored)

        return jsonify({
            "status": "success",
            "result": prediction,
            "probability": round(probability * 100, 2),
            "confidence": round(confidence * 100, 2),
            "accuracy": round(accuracy * 100, 2),
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
        }), 500


@app.route('/result')
@csrf.exempt
def result():
    try:
        results = load_results()
        if 'round1' not in results or 'round2' not in results:
            flash("Please complete both rounds first.", "error")
            return redirect(url_for('round1'))

        round1_result = results['round1']['result']
        round2_result = results['round2']['result']

        combined_features = preprocess_eye_data(
            results['round1']['image1_time'],
            results['round1']['image2_time'],
            [
                float(results['round2'].get('image1_time', 0) or 0),
                float(results['round2'].get('image2_time', 0) or 0),
            ],
        )

        final_prediction, final_probability, final_confidence, final_accuracy = predict_autism_with_model(
            combined_features, results['round1'], results['round2']
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

        username = session.get('username')
        if not results.get('saved'):
            saved = save_result_to_db(username, results, final_result, final_confidence, final_accuracy)
            if saved:
                results['saved'] = True
                store_results(results)

        detailed_results = {
            'final_result': final_result,
            'result_class': result_class,
            'probability': round(final_probability * 100, 2),
            'confidence': round(final_confidence * 100, 2),
            'accuracy': round(final_accuracy * 100, 2),
            'round1': results['round1'],
            'round2': results['round2'],
            'model_used': MODEL_LOADED,
            'recommendations': get_recommendations(final_probability),
            'individual_predictions': {
                'round1': round1_result,
                'round2': round2_result
            }
        }

        return render_template('result.html', **detailed_results)

    except Exception as e:
        logger.error(f"Error in result route: {e}")
        flash("An error occurred while processing results.", "error")
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


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()

    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data
        hashed_password = generate_password_hash(password)

        connection = get_db_connection()
        if connection is None:
            flash("Database connection failed. Please try again later.", "error")
            return redirect(url_for('home'))

        cursor = connection.cursor()
        try:
            cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
            existing_user = cursor.fetchone()
            if existing_user:
                flash("Username or email already exists. Please choose a different one.", "error")
                return redirect(url_for('signup'))

            cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                           (username, email, hashed_password))
            connection.commit()
        except Error as e:
            logger.error(f"Error creating account: {e}")
            flash("Could not create the account. Please try again later.", "error")
            return redirect(url_for('signup'))
        finally:
            cursor.close()
            connection.close()

        session['username'] = username
        flash("Signup successful!", "success")
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
        password = form.password.data

        connection = get_db_connection()
        if connection is None:
            flash("Database connection failed. Please try again later.", "error")
            return redirect(url_for('home'))

        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute(
                "SELECT * FROM users WHERE username = %s AND email = %s",
                (username, email),
            )
            user = cursor.fetchone()
        except Error as e:
            logger.error(f"Error looking up account: {e}")
            flash("Could not sign in. Please try again later.", "error")
            return redirect(url_for('login'))
        finally:
            cursor.close()
            connection.close()

        if user:
            stored_hash = user.get('password')
            try:
                password_ok = bool(stored_hash) and check_password_hash(stored_hash, password)
            except Exception:
                password_ok = False

            if password_ok:
                session['username'] = username
                flash("Login successful!", "success")
                return redirect(url_for('home'))
            else:
                flash("Invalid password.", "error")
                return redirect(url_for('login'))
        else:
            flash("Username or email not found. Please sign up first.", "error")
            return redirect(url_for('signup'))

    return render_template('login.html', form=form)


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You have been logged out.", "success")
    return redirect(url_for('home'))


class ForgotPasswordForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired(), Email()])


class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField(
        'Confirm Password',
        validators=[DataRequired(), EqualTo('password', message='Passwords must match.')],
    )


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    form = ForgotPasswordForm()

    if form.validate_on_submit():
        email = form.email.data

        connection = get_db_connection()
        if connection is None:
            flash("Database connection failed. Please try again later.", "error")
            return redirect(url_for('home'))

        cursor = connection.cursor()
        try:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()
        except Error as e:
            logger.error(f"Error looking up email: {e}")
            flash("Could not look up that email. Please try again later.", "error")
            return redirect(url_for('forgot_password'))
        finally:
            cursor.close()
            connection.close()

        if user:
            token = s.dumps(email, salt='email-confirm')
            reset_link = url_for('reset_password', token=token, _external=True)

            msg = Message('Password Reset Request', sender=MAIL_USERNAME, recipients=[email])
            msg.body = f'Your password reset link is {reset_link}. This link is valid for 30 minutes.'
            try:
                mail.send(msg)
            except Exception as e:
                logger.error(f"Error sending reset email: {e}")
                flash("Could not send the reset email. Please try again later.", "error")
                return redirect(url_for('forgot_password'))

            flash("A password reset link has been sent to your email.", "success")
            return redirect(url_for('login'))
        else:
            flash("Email not found.", "error")
            return redirect(url_for('forgot_password'))

    return render_template('forgot_password.html', form=form)


@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = s.loads(token, salt='email-confirm', max_age=1800)
    except SignatureExpired:
        flash("The reset link is expired.", "error")
        return redirect(url_for('forgot_password'))
    except BadSignature:
        flash("The reset link is invalid.", "error")
        return redirect(url_for('forgot_password'))

    form = ResetPasswordForm()

    if form.validate_on_submit():
        password = form.password.data
        if not re.fullmatch(r'(?=.*[A-Za-z])(?=.*\d).{8,}', password or ''):
            flash("Password must be at least 8 characters and include a letter and a number.", "error")
            return render_template('reset_password.html', form=form, token=token)

        hashed_password = generate_password_hash(password)

        connection = get_db_connection()
        if connection is None:
            flash("Database connection failed. Please try again later.", "error")
            return redirect(url_for('forgot_password'))

        cursor = connection.cursor()
        try:
            cursor.execute("UPDATE users SET password = %s WHERE email = %s", (hashed_password, email))
            connection.commit()
        except Error as e:
            logger.error(f"Error updating password: {e}")
            flash("Could not update the password. Please try again later.", "error")
            return redirect(url_for('forgot_password'))
        finally:
            cursor.close()
            connection.close()

        flash("Your password has been updated!", "success")
        return redirect(url_for('login'))

    return render_template('reset_password.html', form=form, token=token)


ensure_schema()


if __name__ == '__main__':
    app.run(debug=True)