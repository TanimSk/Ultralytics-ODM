import cv2
import numpy as np
import tensorflow.lite as tflite

LABELS = [
    "ripped",
    "stain",
]

# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)


# Load and preprocess image
def preprocess_image(image_path, input_shape):
    print("Input shape:", input_shape)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# [x_min, y_min, x_max, y_max, class_id, confidence]
def postprocess_output(output_data, image_shape):
    image_height, image_width = image_shape
    detections = output_data[0]  # Shape: (6, 8400)
    xc = detections[0]  # (8400,) - normalized center x
    yc = detections[1]  # (8400,) - normalized center y
    w = detections[2]  # (8400,) - normalized width
    h = detections[3]  # (8400,) - normalized height
    confs = detections[4:]  # (8400,) - classes confidence scores

    # confidence threshold
    threshold = 0.5

    boxes = []

    for class_id, conf in enumerate(confs):
        for i in range(len(conf)):
            if conf[i] > threshold:
                # Convert to pixel coordinates
                x_min = int((xc[i] - (w[i] / 2)) * image_width)
                y_min = int((yc[i] - (h[i] / 2)) * image_height)
                x_max = int((xc[i] + (w[i] / 2)) * image_width)
                y_max = int((yc[i] + (h[i] / 2)) * image_height)

                boxes.append([x_min, y_min, x_max, y_max, class_id, conf[i]])

    return boxes


def nms(boxes, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Parameters:
    - boxes: list of lists or np.array of shape (N, 6) -> [x_min, y_min, x_max, y_max, class_id, confidence]
    - iou_threshold: Threshold for Intersection over Union (IoU) to suppress overlapping boxes.

    Returns:
    - A plain Python list of filtered bounding boxes after applying NMS.
    """
    if len(boxes) == 0:
        return []

    # Convert list to NumPy array
    boxes = np.array(boxes)

    # Sort boxes by confidence score in descending order
    boxes = boxes[np.argsort(boxes[:, 5])[::-1]]

    selected_boxes = []

    while len(boxes) > 0:
        # Pick the box with the highest confidence
        chosen_box = boxes[0]
        selected_boxes.append(chosen_box.tolist())  # Convert to list before appending

        # Compute IoU for the remaining boxes
        rest_boxes = boxes[1:]

        x_min = np.maximum(chosen_box[0], rest_boxes[:, 0])
        y_min = np.maximum(chosen_box[1], rest_boxes[:, 1])
        x_max = np.minimum(chosen_box[2], rest_boxes[:, 2])
        y_max = np.minimum(chosen_box[3], rest_boxes[:, 3])

        # Compute intersection area
        intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)

        # Compute areas of the boxes
        chosen_area = (chosen_box[2] - chosen_box[0]) * (chosen_box[3] - chosen_box[1])
        rest_areas = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])

        # Compute IoU
        iou = intersection / (chosen_area + rest_areas - intersection)

        # Keep boxes with IoU below the threshold
        boxes = rest_boxes[iou < iou_threshold]

    return selected_boxes


# Load image
image_path = "image.jpg"
image = cv2.imread(image_path)
image_shape = image.shape[:2]  # (height, width)

# Preprocess image
input_image = preprocess_image(image_path, input_details[0]["shape"])
print("Preprocessed image shape:", input_image.shape)

# Run inference
interpreter.set_tensor(input_details[0]["index"], input_image)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]["index"])
print("Output data shape:", output_data.shape)

# Postprocess output
boxes = postprocess_output(output_data, image_shape)
boxes = nms(boxes, iou_threshold=0.5)
print("Detected objects:", len(boxes))

# Draw bounding boxes
for box in boxes:
    x_min, y_min, x_max, y_max, class_id, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4]), box[5]
    color = (0, 255, 0)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(
        image,
        f"{LABELS[class_id]}: {conf:.2f}",
        (x_min, y_min - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )

# show image
cv2.imshow("image", image)
cv2.waitKey(0)
