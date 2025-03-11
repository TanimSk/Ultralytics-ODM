from tflite_support import metadata
import tensorflow as tf
import numpy as np
import cv2

# Load the TFLite model
model_path = "<model-name>.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print("Input:", input_details)
print("\n\n")
print("Output:", output_details)
print("\n\n")


# ------- Extract metadata from the TFLite model -------
# Load the model file
with open(model_path, "rb") as f:
    model_content = f.read()

# Create a metadata displayer
displayer = metadata.MetadataDisplayer.with_model_file(model_path)

# Get model metadata in JSON format
model_metadata = displayer.get_metadata_json()

# Print metadata
print("TFLite Model Metadata:")
print(model_metadata)
