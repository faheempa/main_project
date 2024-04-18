from flask import render_template, url_for, flash, redirect, request, Blueprint, abort, Response
from project import db, bcrypt
from project.signup_and_login.forms import (
    RegistrationForm,
    LoginForm,
    UpdateAccountForm,
)
from project.models import User
from flask_login import login_user, current_user, logout_user, login_required
from project.signup_and_login.forms import RequestResetForm, ResetPasswordForm
from project.signup_and_login.utils import save_picture, send_reset_email
from project.models import Mock

signup_and_login = Blueprint("signup_and_login", __name__)


@signup_and_login.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))

    form = RegistrationForm()
    if request.method == "GET":
        return render_template(
            "signup_and_login/register.html", title="Register", form=form
        )
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode(
            "utf-8"
        )
        user = User(
            username=form.username.data, email=form.email.data, password=hashed_password
        )
        db.session.add(user)
        db.session.commit()
        flash("Your account has been created! You are now able to log in", "success")
        return redirect(url_for("signup_and_login.login"))
    
    flash(
        "Registration Unsuccessful. Please check your details and try again", "danger"
    )

    return render_template(
        "signup_and_login/register.html", title="Register", form=form
    )


@signup_and_login.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))
    form = LoginForm()
    
    if request.method == "GET":
        return render_template("signup_and_login/login.html", title="Login", form=form)
        
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            next_page = request.args.get("next")
            flash("Login Successful ", "success")
            return redirect(next_page) if next_page else redirect(url_for("main.home"))
        else:
            if user == None:
                flash("Login Unsuccessful. User does not exist", "danger")
            else:
                flash("Login Unsuccessful. Wrong password", "danger")
                
    return render_template("signup_and_login/login.html", title="Login", form=form)


@signup_and_login.route("/logout")
def logout():
    logout_user()
    flash("You have been logged out", "success")
    return redirect(url_for("main.home"))


@signup_and_login.route("/account", methods=["GET", "POST"])
@login_required
def account():
    if current_user.username == "admin":
        abort(403)
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash("Your account has been updated!", "success")
        return redirect(url_for("signup_and_login.account"))
    elif request.method == "GET":
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for("static", filename="img/dp/" + current_user.image_file)
    mocks = Mock.query.filter_by(UID=current_user.id).all()
    return render_template(
        "signup_and_login/account.html",
        title="Account",
        image_file=image_file,
        form=form,
        mocks=mocks,
        balance=current_user.balance,
    )


@signup_and_login.route("/reset_password", methods=["GET", "POST"])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))

    form = RequestResetForm()
    
    if request.method == "GET":
        return render_template("signup_and_login/reset_request.html", title="Reset Password", form=form)
    
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user == None:
            flash("Email does not exist", "danger")
            return redirect(url_for("signup_and_login.reset_request"))
        send_reset_email(user)
        flash("An email has been sent with a link to reset your password.","info",)
        return redirect(url_for("signup_and_login.login"))
    
    return render_template("signup_and_login/reset_request.html", title="Reset Password", form=form)


@signup_and_login.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))
    
    user = User.query.filter(User.password.like(f"{token}%")).first()
    if user is None:
        flash("That is an invalid or expired token", "warning")
        return redirect(url_for("signup_and_login.reset_request"))
    
    form = ResetPasswordForm()
    if request.method == "GET":
        return render_template(
            "signup_and_login/reset_token.html", title="Reset Password", form=form
        )
        
    if form.validate_on_submit():
        new_hashed_password = bcrypt.generate_password_hash(form.password.data).decode(
            "utf-8"
        )
        user.password = new_hashed_password
        db.session.commit()
        flash("Your password has been updated! You are now able to log in", "success")
        return redirect(url_for("signup_and_login.login"))
    
    return render_template(
        "signup_and_login/reset_token.html", title="Reset Password", form=form
    )

@signup_and_login.route("/account/save_amount/<int:amount>")
@login_required
def save_amount(amount):
    current_user.balance = amount
    db.session.commit()
    return Response(status=204)