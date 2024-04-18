from project import create_app, db

app = create_app(db_url="sqlite:///database.db")
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)