from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from project.config import Config

app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "signup_and_login.login"
login_manager.login_message_category = "info"

def create_app(config_class=Config, db_url=None):
    app = Flask(__name__)
    app.config.from_object(Config)

    if db_url:
        app.config['SQLALCHEMY_DATABASE_URI'] = db_url

    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)

    from project.signup_and_login.routes import signup_and_login
    from project.main.routes import main
    from project.errors.handlers import errors
    from project.admin.routes import admin

    app.register_blueprint(signup_and_login)
    app.register_blueprint(main)
    app.register_blueprint(errors)
    app.register_blueprint(admin)

    with app.app_context():
        db.create_all()
        db.session.commit()

    return app