import cv2
import os
import easyocr
from ultralytics import YOLO
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


# Paths to the files and folders
image_path = './Test_video/Vehicle.jpg'
output_folder = './Plate_photo'
model = './Model/Number_plate_recognize_last .pt'


# Load the YOLOv8 model (make sure to use the correct model file or model name)
model = YOLO(model)  # Replace 'yolov8n.pt' with your custom model if needed

image = cv2.imread(image_path)

# Perform detection
results = model(image)

# Initialize EasyOCR reader for both Nepali (Devnagari script) and English
reader = easyocr.Reader(['ne', 'en'])

# Process the results
plate_counter = 0
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf.item()
        class_id = int(box.cls.item())

        # Assuming 'license plate' is the class with index 0
        if class_id == 0 and confidence > 0.5:  # Adjust confidence threshold if needed
            # Crop the detected license plate
            cropped_plate = image[y1:y2, x1:x2]

            # Save the cropped license plate image
            plate_filename = os.path.join(output_folder, f'plate_{plate_counter}.png')
            cv2.imwrite(plate_filename, cropped_plate)
            plate_counter += 1

            # Use EasyOCR to detect text in both Devnagari and English
            ocr_results = reader.readtext(cropped_plate, detail=1)

            for (bbox, text, prob) in ocr_results:
                # Detect if the text is Devnagari
                if any('\u0900' <= char <= '\u097F' for char in text):
                    number_plate = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
                    print(f'Detected Nepali Text: {text}')
                    print(f'Transliterated to English Roman: {number_plate}')
                else:
                    number_plate = text
                    print(f'Detected English Text: {text}')

            # Draw bounding box and label on the image (optional)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'License Plate : {number_plate}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the result (optional)
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


