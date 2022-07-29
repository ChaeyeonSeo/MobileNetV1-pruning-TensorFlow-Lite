# MobileNetV1-pruning-TensorFlow-Lite

- /models: .tflite files and base MobileNetv1 file
- deploy_tflite.py: Deploy tflite file on RPi
- deploy_tflite_multithread.py: Deploy tflite file on RPi using multithreading

```python deploy_tflite.py --model_type [model_type] --file [output_file_suffix]```

- deploy_tflite_armnn.py: Deploy tflite file on RPi using ARMNN (https://developer.arm.com/documentation/102561/2111/Overview-of-running-the-application/Run-the-application?lang=en)

``` python3 deploy_tflite_armnn.py --model_type [model_type] --file [output_file_suffix] --delegate_path [ARMNN library path]```

- main.py: Train models using TensorFlow
- measure.py: Measure temperature and energy usage of RPi
- mobilenet_rm_filt_tf.py: Used for quantization_tf.py
- mobilenet_tf.py: MobileNetv1 code using TensorFlow
- pruning_tf.py: Prune and convert Tensorflow models
- quantization_tf.py: