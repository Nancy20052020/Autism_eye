import os

db_config = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'port': int(os.environ.get('DB_PORT', '3306')),
    'password': os.environ.get('DB_PASSWORD', '20052020'),
    'database': os.environ.get('DB_NAME', 'autism_detection'),
}

SECRET_KEY = os.environ.get('SECRET_KEY', 'your_secret_key')

MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
MAIL_PORT = int(os.environ.get('MAIL_PORT', '587'))
MAIL_USERNAME = os.environ.get('MAIL_USERNAME', 'nancy2005nov@gmail.com')
MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', 'ntgo jlmo jffs geml')
MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() == 'true'
MAIL_USE_SSL = os.environ.get('MAIL_USE_SSL', 'false').lower() == 'true'

MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autism_eye_model.h5'),
)
