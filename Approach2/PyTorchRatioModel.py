import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class RatioNet(nn.Module):
    def __init__(self):
        super(RatioNet, self).__init__()
        self.inlayer = nn.Linear(4, 32)
        self.hidlayer = nn.Linear(32, 64)
        self.hid2layer = nn.Linear(64, 128)
        self.outlayer = nn.Linear(128, 1)  

    def forward(self, x):
        x = F.relu(self.inlayer(x))
        x = F.relu(self.hidlayer(x))
        x = F.relu(self.hid2layer(x))
        x = self.outlayer(x)
        return x

def load_and_process_data(ratios_file, orientations_file):
    """
    Load data from the two files and merge them based on image titles.
    Returns merged dataframe with ratios and orientations.
    """
    ratios_df = pd.read_csv(ratios_file)
    orientations_df = pd.read_csv(orientations_file)
    
    # Merge the dataframes based on image title
    merged_df = pd.merge(ratios_df, orientations_df, on='image_title')
    return merged_df

def prepare_data(merged_df, test_size=0.2, random_state=42):
    """
    Split the data into training and test sets.
    Convert data to PyTorch tensors.
    """
    # Separate features (ratios) and target (orientation)
    X = merged_df[['ratio1', 'ratio2', 'ratio3', 'ratio4']].values
    y = merged_df['orientation'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train, X_test, y_train, y_test

def train_model(model, data, labels, epochs=20000, lr=0.0005):
    """
    Train the neural network model.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def save_model(model, save_path= 'C:\\Users\\ronan.engel\\OneDrive - West Point\\AY 25-1\\CY388\\Actions On\\COA2\\ratio_net_model.pth'):
    """
    Save the trained model's state dictionary.
    
    Args:
        model (nn.Module): The trained PyTorch model
        save_path (str): Path where the model will be saved
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the model state dictionary
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}")
            

def main(ratios_file, orientations_file, model_save_path='C:\\Users\\ronan.engel\\OneDrive - West Point\\AY 25-1\\CY388\\Actions On\\COA2\\ratio_net_model.pth'):
    """
    Main function to run the entire process.
    """
    # Load and process data
    print("Loading data...")
    merged_data = load_and_process_data(ratios_file, orientations_file)
    
    # Prepare data
    print("Preparing data splits...")
    X_train, X_test, y_train, y_test = prepare_data(merged_data)
    
    # Initialize and train model
    print("Training model...")
    model = RatioNet()
    train_model(model, X_train, y_train)
        
    # Save the trained model
    print("Saving model...")
    save_model(model, model_save_path)


ratios_file = "C:\\Users\\ronan.engel\\OneDrive - West Point\\AY 25-1\\CY388\\Actions On\\COA2\\a.csv"  # Expected columns: image_title, ratio1, ratio2, ratio3, ratio4
orientations_file = "C:\\Users\\ronan.engel\\OneDrive - West Point\\AY 25-1\\CY388\\Actions On\\COA2\\gps_mag_image_data_SWEEP6_orientations.csv"  # Expected columns: image_title, orientation
main(ratios_file, orientations_file)