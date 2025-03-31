# Master's thesis 2025

**Malignant Melanoma Diagnosis with Classification Convolutional Neural Networks**

**Elena Georgieva Lecheva**

Student Number: 202007457

MSc in Business Intelligence, June 2025

Supervisor: Julie Norlander

Aarhus University

Aarhus School of Business and Social Sciences
<p></p>
<br>


## Libraries used
This code script for the thesis uses Python 3.11.11 and the following libraries and their versions:
-	pandas 2.2.2
-	numpy 1.26.4
-	os
-	random
-	PIL 11.1.0
-	matplotlib 3.10.0
<p></p>
<br>

## Code explanantion
The overall structure of the code is: import libraries, load data and pre-process data, training, testing, plotting results.

The following text briefly outlines the main functions used to pre-process the data, construct, run and evaluate the models and as well as the motivation behind the choices.
<p></p>
<br>

### Metadata pre-processing
For the pre-processing of the data, I first inspect the metadata. All age and sex values are valid and there are no errors.<br>
Since the missing values are represented by the value “unknown”, I replace “unknown” with “nan” so that python can recognize them as missing.
The imputation of the missing values in age and sex in the training and validation set is based on the ground truth. The missing sex values are imputed as male if the person has melanoma and otherwise with female.
The missing age values are imputed with the age with the highest number of melanoma cases if the person has melanoma, and with the age with the lowest number of melanoma cases if the person does not have melanoma.
Since for the training set we have two ages with the same number of melanoma cases - 65 and 70, I will impute the missing values with the higher age, i.e. 70. For the validation set, this is also 70.
For the people that do not have melanoma, I will impute the missing values with the age that has the most benign cases - 45 for the training set and 15 for the validation set.
Since the training data has one more age group than the validation and the test set, namely age 5, I relabel all ages 5 as age 10 to ensure consistency in the metadata across the 3 sets. This can potentially affect the validity of the model, but since there are only 4 people aged 5 (i.e. 4/2000 = 0.2%), this will not significantly affect the result.
After the imputation, the metadata variables are one-hot encoded, which gives us 18 dummy variables - 2 for sex (male and female) and 16 for age (age 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85).
The metadata for each set is combined with the ground truth, i.e. the label, and the corresponding image in a dataframe.
For each dataframe, the image path and a .jpg file extension is added to each image. In this way the model can access the images during training and inference.
<p></p>
<br>

### Image branch (CNN)
As a base model, I use ResNet50 and ResNet101 with the weights pre-trained on ImageNet by specifying weights = ‘imagenet’. I do not include the fully connected layers so I select include_top = False. This is because I want to use the convolutional base only, i.e. the pre-trained model, to do feature engineering on the input images and then feed them into a new dense fully connected classifier. For this reason, I freeze the base layers. However, as I also fine-tune the network by allowing the weight of some of its layers to be updated, I unfreeze the last 10 layers by setting them to trainable = True.
The preprocess_input function in Keras is specifically designed to prepare the image data for input into the ResNet pre-trained model. It performs preprocessing operations that match the preprocessing steps used during the training of these models. The function converts the input images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
For the head of the model, i.e. the customized dense fully connected classifier which is to be trained together with the unfrozen layers, I used a global average pooling layer. The choice of global average pooling over for flattening is because unlike flattening, global average pooling does not keep positional information. We can argue that in the classification of images of skin lesions the positional information is not important. The only thing that is important is whether the model recognizes the class of the lesion in the image, and not whether it can recognize where the lesion is positioned in the image or whether it can count the number of lesion objects in the images.
After the global average pooling layer, several densely connected layers are added (see the architectures in section Experiments). The shape of the network should be either a tunnel, i.e. the number of neurons in each layer is the same, or a funnel, i.e. the number of neurons in early layers is larger than the one in later layers. However, the shape should not be a bottleneck, i.e. the number of neurons in early layers should not be smaller than the one in later layers. In case of a bottleneck, the model cannot compress all of the information from the data as the representations are too low-dimensional. In our case, the network has a funnel shape.
All layers have use a ReLU activation function for learning complex patterns. A dropout layer to control for overfitting was also included with a dropout rate of 0.5 or 0.3 (see section Experiments).
This network serves as general-purpose feature extractor and a backbone for the classification task. For this reason, no classification function such as sigmoid or softmax is used.
<p></p>
<br>

### Metadata branch
The metadata branch is a basic dense NN, i.e. a simple ANN or Multi-Layer Perceptron (MLP), where the layers are built sequentially. We don't need to explicitly specify Sequential() because this model is already defined as such using Keras' Functional API (“Input”, “Dense” and “Model”).
I first define an input layer where the input shape is specified, which is the number of metadata dimensions after one-hot encoding - 18 (2 for sex and 16 for age).
The model consists of several fully connected (dense) layers (see section Experiments). It again has a funnel shape and each layer uses a relu activation. The output is not specified yet.
<p></p>
<br>

### Combined network
The outputs of the CNN and the ANN network are concatenated into a single input.
After the concatenation of the CNN and ANN models and before the final output layer, we can define additional layers and dropout regularization layers.
The combined network has 2 or 3 additional fully connected (dense) layers and a ReLU activation. These layers learn complex patterns from the combined features (the output from both the CNN and ANN models). The number of neurons in each hidden layer should be at least as big as the number of outcomes. Other values that can also be used are determined by raising 2 to the power of some number. This raised value has to be either equal to a bigger than the number of outcomes. For example, for 46 outcomes, the minimum number should be 2^6 = 64 > 46, but 2^7 = 128 can also work.  In our case, each hidden layer should have at least 2 neurons.
A dropout layer that randomly drops 30% or 50% (see section Experiments) of the neurons during training is also added.
The output layer is defined, which is a fully connected layer modified to match the number of training classes in the classification task. Thus, it has 1 neuron and sigmoid activation as the classification is binary.
<p></p>
<br>

### Image data preparation
As a first step, all 2000, 150 and 600 super-pixel images in the training, validation and test set, respectively, were removed so that only the jpg images were left in the directories.
To pre-process the image data, I first create an ImageDataGenerator for both the training and validation set. A pre-processing method called scaling in applied to both sets. Since images are represented by pixels with colour values between 0 and 255. When passing images as input to a CNN, these values should be normalized, i.e. rescaled by diving them by 255, to values in the same range, i.e. between 0 and 1, because having input data in different ranges will cause problems for the network (Goodfellow et al. 2016).
Data augmentation is applied only to the training data because its primary purpose is to artificially increase the size and diversity of the training dataset, which helps the model generalize better to unseen data. The validation set is used to evaluate the model’s performance during training. It should reflect the real-world data distribution so that you can accurately measure how well the model will perform on unseen data. Thus, augmentation is active only during training but not during inference.
The fill_mode determines how to fill in pixels that get introduced due to transformations (like rotation, zoom or shift). The options include “constant” (pad with a constant value), “wrap” (copy the top part and put it on the bottom and vice versa, copy the right side and put it on the left and vice versa), “reflect” (fill with a reflection of the image's border), 'nearest' (replicate the border). The different fill modes can be seen in the figure below.

![image](https://github.com/user-attachments/assets/737b4047-ccb3-4863-b3a5-ad9bdca4978f)
![image](https://github.com/user-attachments/assets/db868fd6-6eb6-4230-a02b-bbe46d7d628d)
![image](https://github.com/user-attachments/assets/316c1d47-e552-4a7e-bfbe-ddf8f06a8e44)
![image](https://github.com/user-attachments/assets/ab83cce0-b0f2-4f75-b3ab-86948606a64f)

Figure: fill mode constant, wrap, reflect, nearest (from left to right)
<p></p>
<br>

Then I create the actual image datasets using the flow_from_dataframe method of ImageDataGenerator, specifying the dataframes with the directory paths where the images can be found, the image data as the input, the labels and metadata as the output, the target size of the images, and batch size. Since all images, both within and between the 3 datasets, are of varying size (width and height), they were resized to a common size of 224x224.
The own_train_generator_func and own_validation_generator_func are Python generator functions yielding a tuple of inputs (images and metadata) and targets (the labels).
Then I create the training and validation dataset, which are flat map datasets, to be used for training by using the from_generator method and use the output_signature argument, which specifies the shapes and data types of the inputs and output defined by tf.TensorSpec. The output_signature argument is needed in order to ensure that the output of the custom generator functions matches the expected structure and data types.
<p></p>
<br>

### Loss function
In a dataset with class imbalance (e.g., many more negatives than positives), the model can become biased toward predicting the majority class. The focal_loss function defines a custom loss function based on the focal loss concept, which is designed to address class imbalance in tasks like binary classification (Focal Loss paper).
Alpha is a balancing factor to adjust the importance of positive vs. negative classes (default is 0.25). Gamma is a modulating factor that helps the model focus more on hard-to-classify examples (default is 2.0). The final focal loss is a combination of the binary cross-entropy, alpha factor, and modulating factor. The loss is averaged across all instances in the batch.
I use the focal loss in place of binary crossentropy.
<p></p>
<br>

### Optimizer and learning rate
As an optimizer I use Adam because the model seemed to converge a little bit faster than with RMSprop. However, when it comes to results, both optimizers performed the same.
When fine-tuning a pre-trained model, we want to make smaller updates to the weights so that the representations leaned by the pre-trained model are not destroyed. For this reason, I lowered the learning rate from the default one of 1e-3 to 1e-5 in order to take smaller steps to find the minimum loss value. Furthermore, reducing the learning rate makes the validation loss less wiggly and noisy.
<p></p>
<br>

### Training the model
I train the model by passing the train_dataset and validation_dataset, which yield the training and validation inputs and targets.
A relatively high number of epochs is needed in order to make sure we will find the epoch where the model fits the data. I train the models for 25 to 40 epochs (see section Experiments).

The steps_per_epoch argument defines how many steps (i.e. batches) the model should go through in one epoch. The default value is the number of samples in the training dataset divided by the batch size.
Similar to steps_per_epoch, validation_steps defines how many validation steps (batches) should be run in each epoch during the validation phase. The default value is the number of samples in the validation dataset divided by the batch size.
If steps_per_epoch and validation_steps are omitted, TensorFlow assumes that the generator will eventually raise a StopIteration exception, which happens when the generator naturally ends (e.g., iterating through all batches in your dataset).
However, for infinite generators, like ours, this can lead to an infinite training loop because the generator will never stop unless you explicitly define the steps - it will keep calling next() on the generator indefinitely. For this reason, when passing an infinitely repeating dataset, I specify these arguments, otherwise the training or validation will run indefinitely.
If the total number of samples in the datasets is not a perfect multiple of the batch size, Keras will still process the dataset in full, but the last batch may contain fewer samples than the specified batch size. For example, if you have 105 samples in your training dataset and a batch size of 32, Keras will process 3 full batches of 32 samples each (96 samples total), and the last batch will have the remaining 9 samples. The same applies to the validation_steps.
However, since I have already specified the expected shape of the input in output_signature, which includes the batch size, the fit function cannot accept batches of other size. For this reason, the batch size must be an exact divisor of the training, validation and test set, i.e. of 2000, 150 and 600. These numbers are 1, 2, 5, 10, 25, 50. Selecting a good batch size is a matter of finding the right training configuration. However, we want a batch size big enough so that the model can be exposed to enough samples to learn the data patterns. Besides, bigger batches lead to gradients that are more informative and less noisy (lower variance) (Chollet). Based on this, I picked a batch size of 50.
I also use a callback list with a ModelCheckpoint callback to save the model with the lowest validation loss so that it can be loaded later to be used for prediction.
<p></p>
<br>

### Test image data
For the test images, I again create an ImageDataGenerator without data augmentation. Then I create the actual image dataset using the flow_from_dataframe method of ImageDataGenerator, specifying the dataframe with the directory paths where the images can be found, the image data as the input, the labels and metadata as the output, the target size of the images and batch size, the last two being the same as the ones for the training and validation data (224x224 width x height, and 50 images per batch). (For pre-processing of the test metadata, refer to section Metadata).
Then I create own_test_generator_func, similar to the other custom generator fucntions, which is a Python generator function yielding a tuple of inputs (images and metadata) and targets (the labels).

Then I create the test dataset, which is again a flat map datasets, by using the from_generator method and the output_signature argument, which specifies the shapes and data types of the inputs and output defined by tf.TensorSpec, which are the same as for the training and validation data.
For the evaluate method, I again specify the number of steps (i.e. number of batches) that Keras will use to evaluate the model because the test dataset created is also an infinitely repeating dataset. Since output_signature has already specified the expected shape of the input which includes the batch size, the evaluate function cannot accept batches of other size. Therefore, the batch size must be an exact divisor of the training, validation and test set, for which I again choose 50. 
<p></p>
<br>

## References and acknolwedgements
The pre-processing of the data, the model architecture, the custom generator functions and the focal loss function are adapted from the following notebook by Narek HM found on Kaggle:
<p>https://www.kaggle.com/code/nhm1440/image-metadata-with-keras-imagedatagenerator</p>
<br>

The training and validation loss and accuracy plots as well as the custom-defined function "print_best_val_metrics" have been borrowed from exrcise ML4BI_E5_solution from the course Machine Learning for Business Intelligence 2:
<p>https://colab.research.google.com/drive/1pjX4cPtXvnbjSeasNf0Z4Sl-Ii-jPM0L?usp=sharing</p>
<br>

In addition, the Keras and TensorFlow documentation has been used:

Keras documentation: https://keras.io

TensorFlow documentation: https://www.tensorflow.org/

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.
