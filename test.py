import tensorflow as tf

# Load the h5 model
h5_model = tf.keras.models.load_model('cry_detection_model.h5')

# Convert the h5 model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(h5_model)
tflite_model = converter.convert()

# Save the converted TensorFlow Lite model to disk
with open('cry_detection.tflite', 'wb') as f:
    f.write(tflite_model)

import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='my_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set input data
input_data = ... # Load your input data here
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output data
output_data = interpreter.get_tensor(output_details[0]['index'])
