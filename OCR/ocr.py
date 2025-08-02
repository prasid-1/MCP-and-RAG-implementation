
import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import subprocess
import re
import json
from PIL import Image
import pytesseract




def main():
    image_path = 'OCR/images/'
    image_name = 'image3.png'  
    output_path = 'OCR/outputimg/'

    combined_in_path = os.path.join(image_path, image_name)
    combined_out_path = os.path.join(output_path, image_name)

    extracted_text = run_tesseract_ocr(combined_in_path)
    get_json(extracted_text,image_name.strip(".png")+"output_tesseract.json")
    extracted_text = run_easy_ocr(combined_in_path)
    get_json(extracted_text,image_name.strip(".png")+"output_easyOCR.json")

def run_tesseract_ocr(image_path):
    extracted = pytesseract.image_to_string(Image.open(image_path))
    print(extracted)
    return extracted


def run_easy_ocr(combined_in_path):
    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.readtext(combined_in_path)
    # print(result)4

    font = cv2.FONT_HERSHEY_SIMPLEX


    img = cv2.imread(combined_in_path)
    spacer = 100
    completeText = ""
    for detection in result: 
        top_left = (int(detection[0][0][0]), int(detection[0][0][1]))
        bottom_right = (int(detection[0][2][0]), int(detection[0][2][1]))
        text = detection[1]
        img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
        # img = cv2.putText(img,text,(20,spacer), font, 0.5,(0,0,0),1,cv2.LINE_AA)
        spacer+=15
        completeText += text + " "

    print(completeText)

    
    # plt.imshow(img)
    # plt.show()    
    return completeText



def get_json(extracted_text, dir):

    command = ['python', 'OCR/aifilter.py', extracted_text]   

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Target script output:")

        # print(result.stdout)

        if result.stderr:
            print("Target script error output:")
            print(result.stderr)

        
    except subprocess.CalledProcessError as e:
        print(f"Error executing target script: {e}")
        print(f"Stderr: {e.stderr}")

    if result.stdout:
        # Process the output to extract JSON
        json_formatting(result.stdout, dir)
    else:
        print("No output from the target script.")

def json_formatting(output, dir):
    cleanOutput = clean__output(output)

    print(cleanOutput)

    extractOutput = cleanOutput
    
    if extractOutput.startswith("```json"):
        extractOutput = extractOutput[7:]
    if extractOutput.endswith("```"):
        extractOutput = extractOutput[:-3]

    try:
        data = json.loads(extractOutput)
    except json.JSONDecodeError as e:
        print("JSON parsing error:", e)
        exit(1)

    jsonDir = os.path.join("OCR/jsonDataOut/", dir)

    with open(jsonDir, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(" JSON saved to ",jsonDir)


def clean__output(text):
        """Removes the <think>...</think> blocks from DeepSeek R1 output."""
        # This regex matches anything between <think> and </think> (non-greedy)
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned_text.strip() # Remove leading/trailing whitespace


if __name__ == "__main__":
    main()


