from flask import (
    Response,
    render_template,
    Blueprint,
    flash,
    redirect,
    url_for,
    request,
    abort,
)
from flask_login import current_user, login_required
from project.models import Questions, QuestionsAnswered, Mock
from project import db
from project.utils import get_agent, get_action
import random

main = Blueprint("main", __name__)

agent = None
state = None
leverage = 3

@main.route("/")
@main.route("/home")
def home():
    return render_template("main/home.html", title="Home")


@main.route("/about")
def about():
    return render_template("main/about.html", title="About")


@main.route("/trade")
# @login_required
def trade():
    _2022 = [("Mar", 31), ("Apr", 30), ("May", 31), ("Jun", 30), ("Jul", 31), ("Aug", 31), ("Sep", 30), ("Oct", 31), ("Nov", 30), ("Dec", 31)]
    _2023 = [("Jan", 31), ("Feb", 28), ("Mar", 31), ("Apr", 30), ("May", 31), ("Jun", 30), ("Jul", 31), ("Aug", 31), ("Sep", 30), ("Oct", 31), ("Nov", 30), ("Dec", 31)]
    _2024 = [("Jan", 31), ("Feb", 29), ("Mar", 18)]
    calender = [("2022", _2022), ("2023", _2023), ("2024", _2024)]
    total_days = 0
    for year in calender:
        for month in year[1]:
            total_days += month[1]

    new_agent, new_state = get_agent()
    global agent, state
    agent = new_agent
    state = new_state

    # truncate Mock tables's data
    Mock.query.delete()
    db.session.commit()
    
    return render_template("main/topics.html", title="Topics", calender=calender, index=0)

@main.route("/predict/<string:date>")
@login_required
def predict(date):
    global agent, state
    amount = round(0.1 * current_user.balance)
    action, new_state, profit = get_action(agent, state, amount)
    action_type = "buy" if action == 0 else "sell" if action == 1 else "hold"
    profit = round(profit.item()*leverage)
    current_user.balance  = round(current_user.balance + profit)
    data = Mock(Date=date, UID=current_user.id, Amount=amount, Return=profit, Type=action_type)
    db.session.add(data)
    db.session.commit()
    state = new_state
    return {"action": action}

@main.route("/questions/<string:section>/<string:topic>")
@login_required
def questions(section, topic):
    s, t = section.replace("_", " "), topic.replace("_", " ")
    questions = Questions.query.filter_by(Section=s, Topic=t).all()
    answers = QuestionsAnswered.query.filter_by(UID=current_user.id).all()
    answers = {x.QID: (x.Option, x.Status) for x in answers}
    return render_template(
        "main/questions.html",
        title=t,
        questions=questions,
        count=1,
        len=len(questions),
        answers=answers,
    )


@main.route("/save/<string:qid>/<string:status>/<string:selection>")
@login_required
def save(qid, status, selection):
    uid = current_user.id

    data = QuestionsAnswered.query.filter_by(QID=qid, UID=uid).first()
    if data != None:
        data.Status = status
        data.Option = selection
    else:
        data = QuestionsAnswered(
            UID=current_user.id, QID=qid, Status=status, Option=selection
        )

    db.session.add(data)
    db.session.commit()
    return Response(status=204)


@main.route("/mocktest")
@login_required
def mocktest():
    return render_template("main/mocktest.html", title="Mock Test")


@main.route("/mocktest/<string:level>")
@login_required
def mocktest_level(level):
    questions = (
        Questions.query.filter_by(Level=level)  
        .order_by(db.func.random())
        .limit(10)
        .all()
    )
    return render_template(
        "main/mocktest_questions.html",
        title=f"Mock Test - {level}",
        questions=questions,
        count=1,
        len=len(questions),
    )
