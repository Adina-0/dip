from ultralytics import YOLO
import cv2
import os

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # use 'yolov8n.pt' or custom-trained?

# define the path to input directory
input_path = "Data2/4. Bougainvillea"

# Predict and save annotated results
results = model(input_path)

# Process each result for further use
output_folder = "annotated_outputs"
os.makedirs(output_folder, exist_ok=True)

for idx, result in enumerate(results):
    # Get the original image
    image = cv2.imread(result.path)

    # Loop through each detected object
    for box in result.boxes:
        # Extract bounding box coordinates (x1, y1, x2, y2) and confidence
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = int(box.cls[0])  # Class ID

        # Optional: Filter by confidence threshold (e.g., 0.5)
        if confidence < 0.5:
            continue

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{model.names[class_id]}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the annotated image
    output_path = os.path.join(output_folder, f"annotated_{os.path.basename(result.path)}")
    cv2.imwrite(output_path, image)

print(f"Annotated images saved to {output_folder}")