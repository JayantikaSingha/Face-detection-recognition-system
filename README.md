# Face-detection-recognition-system
This project implements a real-time face detection and recognition system using deep learning and machine learning techniques. It utilizes an SSD-based face detector (Caffe) to accurately locate faces in images and video streams, followed by feature extraction using the OpenFace deep neural network to generate discriminative facial embeddings.

These embeddings are classified using a Support Vector Machine (SVM), with hyperparameters optimized via GridSearchCV to ensure improved recognition accuracy. The system supports both dataset-based training and live webcam recognition, displaying predicted identities along with confidence scores and real-time FPS metrics.

This end-to-end solution demonstrates practical experience in computer vision, model training, optimization, and real-time deployment.
