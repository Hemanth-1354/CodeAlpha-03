                                       Handwritten Character Recognition

* This project focuses on building a Convolutional Neural Network (CNN) for recognizing handwritten characters using the EMNIST dataset.


Project Overview:
        The goal of this project is to develop a machine learning model that can identify handwritten letters (A-Z). The model is trained using a CNN architecture, which processes images of handwritten characters and classifies them into one of 26 categories (letters A-Z).


Features:
        Data Preprocessing:
            - The EMNIST dataset was used, where the images were    normalized and reshaped for optimal input into the CNN.
        CNN Architecture:
            - Three convolutional layers followed by max pooling.
            - Fully connected layers with dropout for regularization.
            - Softmax output layer for classification.
        Model Training:
            - The model was trained with Adam optimizer and categorical cross-entropy loss to achieve accurate results.
        Evaluation:
            - The model's performance was evaluated on a separate test set to measure accuracy.


Technologies Used:
    - TensorFlow
    - Keras
    - Python
    - Matplotlib (for visualization)