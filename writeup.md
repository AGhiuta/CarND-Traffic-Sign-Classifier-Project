#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data_sample.png "Data Sample"
[image2]: ./examples/bar_chart.png "Bar Chart"
[image3]: ./examples/grayscale.png "Grayscale"
[image4]: ./test_images/1.png "Traffic Sign 1"
[image5]: ./test_images/2.png "Traffic Sign 2"
[image6]: ./test_images/3.png "Traffic Sign 3"
[image7]: ./test_images/4.png "Traffic Sign 4"
[image8]: ./test_images/5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/AGhiuta/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for the summary of the data set is contained in the third code cell of the IPython notebook.  

I used the **numpy** library to calculate summary statistics of the traffic signs data set:

* The size of training set is **33176**
* The size of test set is **10368**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in code cells 3-8 of the IPython notebook.

First, I display a sample of images for each class. This is done in code cell 6. Here is what it looks like for the first 4 classes of the data set:

![alt text][image1]

Then I compute and display (as a bar chart) the number of images per class. This is done in code cell 8, and the result is shown below:

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in code cells 12-14 of the IPython notebook.

The first step of preprocessing consisted in converting the RGB image to YUV, followed by applying local contrast equalization to the luminance (Y) channel. This is done to improve the contrast and to make all the images with same lighting conditions. The result can be seen below: 

![alt text][image3]

As a last step, I normalized the UV channels in order to bring them in the same range as Y.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the second code cell of the IPython notebook.

I combined all samples into a single dataset, then I used sklearn's **train_test_split** to slice out 20% for test data and 20% for validation data.

Code cells 9-11 of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because the data summarization process revealed that the data wasn't balanced across classes, which would've resulted in a model biased toward the classes containing the most images. The augmentation was done by rotating the images in certain classes, with angles between [-15, 15] degrees.

My final training set had 63712 images. My validation set and test set had 13669 and 10368 images.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Y image     							| 
| Input         		| 32x32x2 UV image     							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x30 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x8 	|
| Concat 				| outputs 28x28x38 								|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x38 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 					|
| Flatten   	      	| input 14x14x38,  outputs 7448 				|
| Flatten   	      	| input 5x5x64,  outputs 1600   				|
| Concat 				| outputs 9048  								|
| Fully connected		| outputs 100  									|
| Dropout       		|												|
| Fully connected		| outputs 100  									|
| Dropout       		|												|
| Fully connected		| outputs 43  									|
| Softmax				|												|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in cells 17, 19-21 of the ipython notebook. 

To train the model, I used a 2-stage approach:
* first, I train the model for 100 epochs, with batch size of 256 and learning rate of 0.001
* then, I train the model for another 10 epochs, with a learning rate of 0.0001; decreasing the learning rate helps the model to become more specialized
    
I used Xavier initializer, which automatically determines the scale of initialization based on the layer dimensions. I used AdamOptimizer for minimizing the cost function.

In order to prevent overfitting, I applied dropout regularization on the fully-connected layers, with a keep probability of .75

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the cells 21-22 of the Ipython notebook.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.996 
* test set accuracy of 0.995

As model architecture, I used the 38-64 model with 2 fully connected layers, described by Pierre Sermanet and Yann LeCun in this paper: [sermanet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

The key feature of this architecture is the skip layer, which combines the low-level features of the first convolution (edges), with the higher-level features of the second convolution (shapes) in order to better classify traffic sign images.

The model's final accuracies on the 3 datasets (trainig, validation and test) show that the model didn't overfit.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The difficulty of the first image consists in the fact that the traffic sign is poorly lit.

The 2nd image might be difficult to classify because the dataset contains several classes that are quite similar (Speed limit 20km/h, Speed limit 30km/h, Speed limit 50km/h, Speed limit 60 km/h, Speed limit 80 km/h), thus making the discrimination between them more difficult.

The same goes for the 3rd image: there are several classes similar to this one (Right-of-way at the next intersection, Road narrows on the right, Traffic signals), which makes the classification harder.

The 4th image might be difficult to classify because it is partially occluded. 

The last image is not part of the dataset on which the model was trained, so it obviously fails on classifying this one. It was also expected the model to be less confident in its predictions on this test case than on the other ones.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			 			            |     Prediction	        			| 
|:-------------------------------------:|:-------------------------------------:|
| Stop						      		| Stop							 		|
| Speed limit (80km/h)					| Speed limit (80km/h)   				| 
| General caution  						| General caution 								|
| No entry	| No entry	|
| No vehicles over 2.1m					| Keep right      						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 28th cell of the Ipython notebook.

For the first 4 images, the model is absolutely sure in its predictions (probability >0.99), and the images do indeed contain the signs it predicted with high confidence.

The last test image (No vehicles over 2.1m) is not part of the dataset on which the model was trained, so it was expected that the model would fail on this image. Also, it was less confident when classifying this image (0.83) than it was when classifying the other ones.


**Image 1 Probabilities:**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Stop 											| 
| .0    				| Keep right 										|
| .0					| No entry       								|
| .0	      			| Turn left ahead  |
| .0				    | No vehicles 									|

**Image 2 Probabilities:**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Speed limit (80km/h)   						| 
| .0     				| Speed limit (100km/h) 						|
| .0					| Speed limit (60km/h)							|
| .0	      			| Speed limit (30km/h)			 				|
| .0				    | Speed limit (50km/h)							|


**Image 3 Probabilities:**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .997         			| General caution       						| 
| .002    				| Right-of-way at the next intersection 		|
| .0					| Road work         							|
| .0	      			| Yield             			 				|
| .0				    | Wild animals crossing							|

**Image 4 Probabilities:**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| No entry 		| 
| .0    				| Speed limit (30km/h)							|
| .0					| Roundabout mandatory      							|
| .0	      			| Turn left ahead 						|
| .0				    | Speed limit (20km/h)									|

**Image 5 Probabilities:**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .834         			| Keep right 									| 
| .096    				| Keep left 									|
| .039					| Go straight or left      						|
| .013	      			| Roundabout mandatory 							|
| .007				    | Go straight or right     						|
