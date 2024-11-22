# Main file
from flask import Flask,render_template
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
@app.route("/")
def hello_world():
    return render_template('index.html')
@app.route("/products")
def products():
    return "<p>This is products page!</p>"
@app.route("/login")
def login():
    return render_template('login.html')
@app.route("/about")
def about():
    return render_template('about.html')
@app.route("/avatar")
def avatar():
    return render_template('Avatar.html')
if __name__=="__main__":
    app.run(debug=True)
