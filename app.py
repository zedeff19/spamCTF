from flask import Flask, render_template, request
from forms import SignupForm, SpamForm  # Ensure both forms are imported
import torch
import torch.nn as nn
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os





app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Update the input size to match the TF-IDF vectorizer output
        self.model = nn.Sequential(
            nn.Linear(44934, 1),  # Updated input size
            # nn.ReLU(),
            # nn.Linear(100, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        perturbation = torch.randn_like(x) * 0.01
        x = x + perturbation
        return self.model(x)

# Load the TF-IDF vectorizer
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the model
model = SimpleModel()
# Load the state dict and remap the keys
state_dict = torch.load('models/model.pth')

# Adjust the keys in the state_dict to match the expected format
new_state_dict = {}
try : 
    new_state_dict['model.0.weight'] = state_dict['0.weight']
    new_state_dict['model.0.bias'] = state_dict['0.bias']
except KeyError as e : 
    try : 
        new_state_dict['model.0.weight'] = state_dict['model.0.weight']
        new_state_dict['model.0.bias'] = state_dict['model.0.bias']
    except Exception : 
        pass
# new_state_dict['model.2.weight'] = state_dict['2.weight']
# new_state_dict['model.2.bias'] = state_dict['2.bias']
model.load_state_dict(new_state_dict)
model.eval()

def encode_string(input_string):
    if input_string is None:
        return torch.tensor([], dtype=torch.float32)
    
    # Transform the text using the TF-IDF vectorizer
    X = vectorizer.transform([input_string])
    print("Shape of TF-IDF features:", X.shape)  # Debugging line
    # Convert the sparse matrix to a dense tensor
    return torch.tensor(X.toarray(), dtype=torch.float32)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['GET', 'POST'])
def check():
    form = SpamForm()  # Use SpamForm here
    if form.is_submitted() and request.method == 'POST':
        # Debugging line to check if emailText is retrieved correctly
        email_text1 = request.form.get('emailText')
        print("emailText:", email_text1)  # Debugging line

        if email_text1:
            email_text = encode_string(email_text1)
            booli = True
            if round(model(email_text).item()) == 0:
                booli = False
            print(booli)
        else:
            email_text1 = ""
            booli = None
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # Save the file to the local file system
                file.save(f'Data/{file.filename}')
        
        return render_template('check.html', email_text=email_text1, form=form, booli=booli)
    return render_template('check.html', form=form)


@app.route('/update_flag', methods=['GET'])
def update_flag():
    # Here you can add logic to handle the flag update if needed
    return "Model training complete and Flask app notified!"

if __name__ == '__main__':
    app.run(debug=True)
