import os
from dotenv import load_dotenv

load_dotenv()

host = os.getenv('DB_HOST')
username = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')
database = os.getenv('DB_DATABASE')

class Config:
    SQLALCHEMY_DATABASE_URI = f"mysql+mysqlconnector://{username}:{password}@{host}:3306/{database}" # for mysql online db
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv('SECRET_KEY')
    MAIL_USERNAME = os.getenv('EMAIL_USER')
    MAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')



