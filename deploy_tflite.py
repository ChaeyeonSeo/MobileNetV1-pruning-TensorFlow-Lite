from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import tflite_runtime.interpreter as tflite
import time
import argparse
import subprocess

parser = argparse.ArgumentParser(description='EE379K HW4 - TFLite deployment')

parser.add_argument('--model_type', type=str, default='mbnv1_frac0.2_ep5', help='Choose between: [VGG11, VGG16, MobileNetv1]')
parser.add_argument('--file', type=str, default='1')
args = parser.parse_args()

model_name = args.model_type
file_name = args.file

measure = subprocess.Popen(f"python measurement.py --file {model_name}_{file_name}", shell=True)

tflite_model_name = './models/' + f'{model_name}.tflite'

# Get the interpreter for TensorFlow Lite model
interpreter = tflite.Interpreter(model_path=tflite_model_name)

# Very important: allocate tensor memory
interpreter.allocate_tensors()

# Get the position for inserting the input Tensor
input_details = interpreter.get_input_details()
# Get the position for collecting the output prediction
output_details = interpreter.get_output_details()

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

inference_time = 0.
cnt = 0

for filename in tqdm(os.listdir("/home/pi/Project/test_deployment")):
  with Image.open(os.path.join("/home/pi/Project/test_deployment", filename)).resize((32, 32)) as img:
    input_image = np.expand_dims(np.float32(img), axis=0)
    input_image = input_image / 255.0

    # Set the input tensor as the image
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run the actual inference
    start_time = time.time()
    interpreter.invoke()
    inference_time += time.time() - start_time

    # Get the output tensor
    pred_tflite = interpreter.get_tensor(output_details[0]['index'])

    # Find the prediction with the highest probability
    top_prediction = np.argmax(pred_tflite[0])

    # Get the label of the predicted class
    pred_class = label_names[top_prediction]
    filename = filename.split('_')
    filename = filename[1].split('.')
    true_class = filename[0]
    if pred_class == true_class:
        cnt += 1

accuracy = cnt / 100
print(f"Inference accuracy: {cnt / 100}%")
print(f"Inference time: {inference_time}")

with open(f'runtime_{model_name}_{file_name}.csv', 'w') as file:
  file.write(f'accuracy,inference_time\n'
             f'{accuracy},{inference_time}\n')

measure.terminate()
