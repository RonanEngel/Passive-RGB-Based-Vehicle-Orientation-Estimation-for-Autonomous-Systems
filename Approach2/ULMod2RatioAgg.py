from ultralytics import YOLO
import numpy
import os
from pathlib import Path

# UL Model Path
ULMod = 'C:\\Users\\ronan.engel\\OneDrive - West Point\\AY 25-1\\CY388\\Actions On\\Models\\side_detector3.pt'

def area_computation(image):
    out_ratios = [0, 0, 0, 0]
    area_sum = 0

    for box in image:
        TLxF, TLyF, BRxF, BRyF, conf, obj_class = box
        area = abs((float(BRxF) - float(TLxF)) * (float(BRyF) - float(TLyF)))
        area_sum += area
        out_ratios[int(obj_class) - 1] += area

    
    if area_sum > 0:
        for i in range(4):
            out_ratios[i] /= area_sum
    
    return out_ratios

def run_model(ULMod, image_dir):
    model = YOLO(ULMod)
    outList = []
    filenames = []
    
    # process full images directory
    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith('.png'):
            image_path = os.path.join(image_dir, image_file)
            
            # run UL model on image
            results = model(image_path)
            
            # comb results to get needed ratio data
            for result in results:
                boxes = result.boxes.cpu().numpy()
                add_on = []
                for obj in boxes.data:
                    add_on.append(obj)
                
                if add_on:  
                    outList.append(area_computation(add_on))
                    filenames.append(image_file)

    return outList, filenames

def CSV_Output_build(filenames, ratio_output, output_path):
    with open(output_path, 'w') as file:
        # write out header
        file.write('filename,ratio1,ratio2,ratio3,ratio4\n')
        
        # write out data
        for filename, ratios in zip(filenames, ratio_output):
            line = f"{filename},{','.join(map(str, ratios))}\n"
            file.write(line)

image_dir = "C:\\Users\\ronan.engel\\OneDrive - West Point\\AY 25-1\\CY388\\Actions On\\Image Folders\\webcam_images_SWEEP6"
output_csv = "C:\\Users\\ronan.engel\\OneDrive - West Point\\AY 25-1\\CY388\\Actions On\\COA2\\a.csv"
    


ratios, filenames = run_model(ULMod, image_dir)
    
CSV_Output_build(filenames, ratios, output_csv)
