from flask import url_for
import os
import secrets
from PIL import Image
from email.message import EmailMessage
import smtplib
import ssl
from project import app


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, "static/img/dp", picture_fn)

    output_size = (100, 100)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn

def send_reset_email(user):
    token = user.password.split("/")[0]
    sender = app.config["MAIL_USERNAME"]
    password = app.config["MAIL_PASSWORD"]
    reciver = user.email
    subject = "Password Reset Request"
    body = f"""To reset your password, visit the following link:
{url_for('signup_and_login.reset_token', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made.
    """
    em = EmailMessage()
    em["From"] = sender
    em["To"] = reciver
    em["Subject"] = subject
    em.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(sender, password)
        smtp.sendmail(sender, reciver, em.as_string())