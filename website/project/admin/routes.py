from flask import render_template, Blueprint, abort
from flask_login import login_required, current_user
from flask import render_template, url_for, redirect, request
from project.admin.forms import AddQuestionForm, UpdateQuestionForm
from project.models import Questions
from project import db

admin = Blueprint('admin', __name__)


@admin.route("/admin")
@login_required
def admin_home():
    if not current_user.username == 'admin':
        abort(403)
    page_number = request.args.get('page', 1, type=int)
    questions = Questions.query.order_by(Questions.QID.desc()).paginate(page=page_number, per_page=10)
    return render_template('admin/admin.html', items=questions, title='Admin Home')


@admin.route("/admin/remove/<int:question_id>")
@login_required
def remove(question_id):
    if not current_user.username == 'admin':
        abort(403)
        
    question = Questions.query.get_or_404(question_id)
    db.session.delete(question)
    db.session.commit()
    return redirect(url_for('admin.admin_home'))

@admin.route("/admin/add_question", methods=['GET', 'POST'])
@login_required
def add_question():
    if not current_user.username == 'admin':
        abort(403)
    form = AddQuestionForm()
    
    if request.method == 'POST':
        question = Questions(QID = Questions.getNextQID(), Question=form.question.data, 
                             Answer=form.answer.data, OptionA=form.option_a.data, 
                             OptionB=form.option_b.data, OptionC=form.option_c.data, 
                             OptionD=form.option_d.data, Section=form.section.data, Topic=form.topic.data)
        db.session.add(question)
        db.session.commit()
        return redirect(url_for('admin.admin_home'))
    
    return render_template("admin/form.html", title='Question Form', form=form)

@admin.route("/admin/update_question/<int:question_id>", methods=['GET', 'POST'])
@login_required
def update_question(question_id):
    if not current_user.username == 'admin':
        abort(403)
        
    question = Questions.query.get_or_404(question_id)
    form = UpdateQuestionForm()
    if request.method == 'GET':
        form.question.data = question.Question
        form.answer.data = question.Answer
        form.option_a.data = question.OptionA
        form.option_b.data = question.OptionB
        form.option_c.data = question.OptionC
        form.option_d.data = question.OptionD
        form.section.data = question.Section
        form.topic.data = question.Topic
        
    if request.method == 'POST':
        question.Question = form.question.data
        question.Answer = form.answer.data
        question.OptionA = form.option_a.data
        question.OptionB = form.option_b.data
        question.OptionC = form.option_c.data
        question.OptionD = form.option_d.data
        question.Section = form.section.data
        question.Topic = form.topic.data
        db.session.commit()
        return redirect(url_for('admin.admin_home'))
    
    return render_template("admin/form.html", title='Form', form=form)