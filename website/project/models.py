from project import db, login_manager
from flask_login import UserMixin
from sqlalchemy import text

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default="default.jpg")
    password = db.Column(db.String(60), nullable=False)
    balance = db.Column(db.Integer, nullable=False, default=0)


class Questions(db.Model):
    QID = db.Column(db.Integer, primary_key=True)
    Question = db.Column(db.String(2000), nullable=False)
    Answer = db.Column(db.String(250), nullable=False)
    OptionA = db.Column(db.String(250), nullable=False)
    OptionB = db.Column(db.String(250), nullable=False)
    OptionC = db.Column(db.String(250))
    OptionD = db.Column(db.String(250))
    Section = db.Column(db.String(250), nullable=False)
    Topic = db.Column(db.String(250), nullable=False)
    Level = db.Column(db.String(250))

    @staticmethod
    def getNextQID():
        val = db.session.execute(text(f"select max(QID) from questions")).all()[0][0]
        return val + 1 if val else 1


class QuestionsAnswered(db.Model):
    UID = db.Column(db.Integer, primary_key=True)
    QID = db.Column(db.Integer, primary_key=True)
    Status = db.Column(db.String(10), nullable=False)
    Option = db.Column(db.String(10), nullable=False)
    
    
class Mock(db.Model):
    Date = db.Column(db.String(25), primary_key=True)
    UID = db.Column(db.Integer)
    Amount = db.Column(db.Integer, nullable=False)
    Type = db.Column(db.String(10), nullable=False)
    Return = db.Column(db.Integer, nullable=False)
