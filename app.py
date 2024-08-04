from flask import Flask, render_template, request
from forms import SignupForm, SpamForm  # Ensure both forms are imported
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
def encode_string(input_string):
    # Example encoding: convert each character to its ASCII value and pad/truncate to length 10
    encoded = [ord(c) for c in input_string[:10]]
    if len(encoded) < 10:
        encoded += [0] * (10 - len(encoded))  # Pad with zeros if less than 10 characters
    return torch.tensor(encoded, dtype=torch.float32)


# Load the model
model = SimpleModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['GET', 'POST'])
def check():
    form = SpamForm()  # Use SpamForm here
    if form.is_submitted() and request.method == 'POST':
        email_text1 = request.form.get('emailText') 
        email_text = encode_string(email_text1) 
        # booli = False # this is the op of our model which is 0 or 1
        booli = False
        booli = model(email_text).item() > 0
        return render_template('check.html', email_text=email_text1, form=form, booli = booli)
    return render_template('check.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
