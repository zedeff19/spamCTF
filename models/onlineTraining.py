import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pickle
import warnings
warnings.filterwarnings('ignore')


# Define your model class (should be the same as your initial model)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(44934, 1),  # Ensure this matches your TF-IDF output size
            nn.Sigmoid()
        )
    
    def forward(self, x):
        perturbation = torch.randn_like(x) * 0.01
        x = x + perturbation
        return self.model(x)

# Load the existing model
model = SimpleModel()
state_dict = torch.load('models/model.pth')
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

model.load_state_dict(new_state_dict)
print(model.state_dict)
model.eval()

# Load your new training data
# data = pd.read_csv('Data/combined_data.csv')
data = pd.read_csv('Data/User_Input.csv')
data = data.head(50)
texts = data['text'].tolist()
labels = data['label'].tolist()

# Initialize the TF-IDF vectorizer and transform the text data
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
X = vectorizer.transform(texts)
y = torch.tensor(labels, dtype=torch.float32)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert sparse matrices to dense tensors
X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_val = torch.tensor(X_val.toarray(), dtype=torch.float32)

# Define Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create DataLoader
train_dataset = TextDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = TextDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Set the model to training mode
model.train()
criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# Continue training the model
num_epochs = 20  # Adjust as needed
for epoch in range(num_epochs):
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss / len(val_loader):.4f}')

# Save the updated model weights
torch.save(model.state_dict(), 'models/model.pth')
print(f"Updated state_dict : {model.state_dict}")