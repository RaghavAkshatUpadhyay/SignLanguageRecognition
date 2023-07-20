# SignLanguageRecognition
Purpose of the Project

The goal of this project is to build a Convolutional Neural Network based recognition system which will be able to classify which letter of the American Sign Language (ASL) alphabet is being signed, given an image of a signing hand. This project is a step towards building a possible sign language translator, which can take communications in sign language and translate them into written and oral language. Such a system would greatly lower the barrier for many deaf and mute individuals to be able to better communicate with others in day-to-day interactions. 

2.1	Reference Algorithm

For this project, we have used one reference algorithms:

•	Convolutional Neural Network: is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a CNN is much lower as compared to other classification algorithms. Initially we plan to create a simple neural network with one convolution layer then moving forward we will apply parameter tuning on the model using multiple layers, Batch Normalization, and Dropout layer.

	Data Collection – This project works upon American Sign Language MNIST dataset which is publicly available on the Kaggle platform. https://www.kaggle.com/datasets/datamunge/sign-language-mnist. The dataset consists of pixel values of hand sign images of various alphabets of the English Language.
	Data Pre-Processing – After the initial phase of data collection, the next task in line was pre-processing the data we received was already in csv format and had pixel values in it. For some models we had to apply augmentation in the data so as to get a better accuracy.
	Model Creation – After pre-processing, we will use the Tensorflow and keras library of python to create our neural network. We will try to make a number of models using different number of convolution layer, adding or removing batch-normalization, augmentation and dropout layer.
	Real Time Detection– If we try to create a Real Time System we will use OpenCV library of python to get input images from our camera.
	Testing and Analysis – This step contains the conclusion, results and observations of the model after proper testing and analysis.

