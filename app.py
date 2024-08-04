from flask import Flask, render_template, request
from forms import SignupForm, SpamForm  # Ensure both forms are imported
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['GET', 'POST'])
def check():
    form = SpamForm()  # Use SpamForm here
    if form.is_submitted() and request.method == 'POST':
        email_text1 = request.form.get('emailText')  
        booli = True
        return render_template('check.html', email_text=email_text1, form=form, booli = booli)
    return render_template('check.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
