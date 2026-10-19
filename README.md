# Autism Eye

Webcam eye-tracking screening app. It times how long someone looks at paired images across two rounds, then estimates whether that gaze pattern is more consistent with autism-related viewing. People can create an account, sign in, and reset a forgotten password.

This is a screening aid for learning and early awareness. It is not a medical diagnosis.

## Setup

1. Use Python 3.10 or newer, and have MySQL running.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional: copy `.env.example` to `.env` and export the variables you want to override. Local defaults already live in `config.py` (database, mail, and secret key). Set `MODEL_PATH` only if you have a trained `autism_eye_model.h5` file. Without that file, the app uses its built-in rule-based estimate.

4. Start the app:

```bash
python app.py
```

5. Open http://127.0.0.1:5000

On startup the app creates the `autism_detection` database and the `users` and `test_results` tables when MySQL accepts the connection.

TensorFlow is optional. Install it only if you want the Keras model path:

```bash
pip install tensorflow
```

## How a test runs

1. Sign up or continue as a guest from the home page.
2. Choose **Start Test**, then Round 1.
3. Allow the camera and click **Quick Setup**. Look at both images, then submit.
4. Round 2 starts on its own. Submit again.
5. The result page shows a low, moderate, or high possibility, plus the time spent on each image.

Use Chrome or Edge. The eye tracker runs in the browser with [WebGazer](https://webgazer.cs.brown.edu/).

## Pages

| Path | Purpose |
| --- | --- |
| `/` | Home |
| `/signup`, `/login`, `/logout` | Account |
| `/forgot_password`, `/reset_password/<token>` | Password reset |
| `/autism` | Test introduction |
| `/round1`, `/round2` | Gaze rounds |
| `/result` | Combined result |

## Configuration

| Variable | Purpose |
| --- | --- |
| `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` | MySQL connection |
| `SECRET_KEY` | Session and reset-link signing |
| `MAIL_SERVER`, `MAIL_PORT`, `MAIL_USERNAME`, `MAIL_PASSWORD` | Password-reset email |
| `MAIL_USE_TLS`, `MAIL_USE_SSL` | SMTP security |
| `MODEL_PATH` | Optional Keras `.h5` model |

Password reset uses the mailbox in `MAIL_USERNAME`. Gmail needs an app password, not the normal account password.
