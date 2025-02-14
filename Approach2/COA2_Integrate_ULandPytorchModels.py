import torch
from ultralytics import YOLO
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


### PATH TO UL MODEL ###
ULMod = 'C:\\Users\\ronan.engel\\OneDrive - West Point\\AY 25-1\\CY388\\Actions On\\Models\\side_detector3.pt'
image_dir = "C:\\Users\\ronan.engel\\OneDrive - West Point\\AY 25-1\\CY388\\Actions On\\Image Folders\\webcam_images_SWEEP6"
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


def area_computation(image):
    out_ratios = [0, 0, 0, 0]
    area_sum = 0

    for box in image:
        TLxF, TLyF, BRxF, BRyF, conf, obj_class = box
        area = abs((float(BRxF) - float(TLxF)) * (float(BRyF) - float(TLyF)))
        area_sum += area
        out_ratios[int(obj_class) - 1] += area

    # Avoid division by zero
    if area_sum > 0:
        for i in range(4):
            out_ratios[i] /= area_sum
    
    return out_ratios


def predict_angle(ratios):
    # Load the pre-trained model
    model = RatioNet()
    
    
    model.load_state_dict(torch.load("C:\\Users\\ronan.engel\\OneDrive - West Point\\AY 25-1\\CY388\\Actions On\\COA2\\ratio_net_model.pth"))
    model.eval()
    
    # Convert ratios to tensor
    input_tensor = torch.tensor(ratios, dtype=torch.float32).unsqueeze(0)
    
    # Predict angle
    with torch.no_grad():
        predicted_angle = model(input_tensor)
    
    return predicted_angle.item()

def run_model(ULMod, image_dir, output_csv):
    model = YOLO(ULMod)
    predicted_angles = []


    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['name', 'Predicted Angle'])
    
        for image_file in os.listdir(image_dir):
            if image_file.lower().endswith('.png'):
                image_path = os.path.join(image_dir, image_file)
                
                results = model(image_path)
                
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    add_on = []
                    for obj in boxes.data:
                        add_on.append(obj)
                    
                    if add_on:  
                        ratios = area_computation(add_on)
                        angle = (predict_angle(ratios) % 360)
                        csv_writer.writerow([image_file, f"{angle:.2f}"])

                        print(f"{image_file}: {angle:.2f} degrees")
                        predicted_angles.append(angle)



    return predicted_angles


output_csv = f'C:\\Users\\ronan.engel\\OneDrive - West Point\\AY 25-1\\CY388\\Actions On\\Testing\\COA2 Testing\\{image_dir[-6:]}_predictions.csv'

predicted_angles = run_model(ULMod, image_dir, output_csv)
print(f"Results Saved")

