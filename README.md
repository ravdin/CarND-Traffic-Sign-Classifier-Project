# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
#### Writeup / README

##### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ravdin/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

#### Data Set Summary & Exploration

##### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic signs data set:

 * The size of training set is 34799
 * The size of test set is 12630
 * The shape of a traffic sign image is 32x32x3
 * The number of unique classes/labels in the data set is 43

##### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

For the exploratory step, I took a random sampling of 8 images from each classification.  I also included a bar chart to show the distribution by class of the images in the training set.

I came to the following conclusions from the exploratory analysis:

* The brightness/contrast of the data set is highly varied and it would probably be helpful to normalize the brightness across the data set in the preprocessing step.
* The colors in the images don't seem to be important, so we can probably benefit from converting to grayscale to emphasize the edges and contours.
* Many classifications seem to be underrepresented.  Without data augmentation, we might not expect new samples to perform well.

#### Design and Test a Model Architecture

##### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because I wanted to emphasize edges and contours to the learner instead of colors.

Next, I used [exposure.adapt_hist](http://scikit-image.org/docs/dev/api/skimage.exposure.html#equalize-adapthist) from the `sklearn` library to equalize the contrast.  The idea is to minimize the extent to which variations in brightness might affect the outcome.

As a last step, I normalized the image data to bring the values to a range of [0, 1].

##### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the training, validation, and test sets as provided in the project instructions.

* Training: 34799 samples
* Validation: 4410 samples
* Test: 12360 samples

##### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the ninth cell of the ipython notebook.  I used a modified LeNet architecture as we used in the lab example for digit recognition.  I made a change to use dropout in the fully connected layers to compensate for overfitting.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 							|
| Flatten           | Output 400        |
| Fully connected		| Output 120        									|
| RELU				|         									|
|	Dropout		  |	50% dropout rate for training											|
|	Fully connected	|	Output 84				|
| RELU				|         									|
|	Dropout		  |	50% dropout rate for training											
| Fully connected		| Output 43        								

##### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eleventh cell of the ipython notebook.

To train the model, I used an Adam Optimizer with a learning rate of 0.001.  I considered regularizing the outputs but I found when experimenting that it was suboptimal.  I used a batch size of 128 and trained for 30 epochs.

##### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 94.8%
* test set accuracy of 93.6%

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I chose the LeNet architecture because it is well tested and proven.  I believed it would be relevant to the traffic signs because it predicts across a relatively small number of classes.  It seems to have performed well in terms of the test set as the validation and test performance are fairly close- this leads me to believe that I successfully avoided overfitting in the model.

### Test a Model on New Images

##### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

[example0]: ./images/class%202.jpg "Speed limit 50"
[example1]: ./images/class%203.jpeg "Speed limit 60"
[example2]: ./images/class%2011.jpeg "Right of way"
[example3]: ./images/class%2038.jpeg "Keep right"
[example4]: ./images/class%2040.jpeg "Roundabout mandatory"

![alt text][example0] ![alt text][example1] ![alt text][example2]
![alt text][example3] ![alt text][example4]

I cropped the images before running them through the model so that the aspect ratio would be reasonably close to 1:1.

##### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the first cell of the Ipython notebook after Step 3.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit 50     		| Speed limit 50   									|
| Speed limit 60     			| Speed limit 60 										|
| Right of way					| Right of way											|
| Keep right      		| Keep Right					 				|
| Roundabout Mandatory		| Priority Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

For the final image I noted that there is a watermark, which might have thrown off the model (and would not be realistic).  The model did not make a confident prediction, as seen below.

##### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the final cell of the Ipython notebook.

#### Image 0: Speed limit 50

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|Speed limit (50km/h): |0.80|
|Speed limit (80km/h): |0.14|
|Speed limit (60km/h): |0.05|
|Speed limit (30km/h): |0.01|
|Speed limit (70km/h): |0.00|


#### Image 1: Speed limit 60

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|Speed limit (60km/h): |0.73
|Speed limit (80km/h): |0.20
|Speed limit (50km/h): |0.08
|Speed limit (30km/h): |0.00
|Keep right: |0.00

#### Image 2: Right of way
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|Right-of-way at the next intersection: |0.99
|Beware of ice/snow: |0.00
|Pedestrians: |0.00
|Double curve: |0.00
|Roundabout mandatory: |0.00

#### Image 3: Keep right
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|Keep right: |1.00
|Yield: |0.00
|Stop: |0.00
|Slippery road: |0.00
|Turn left ahead: |0.00

#### Image 4: Roundabout mandatory
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|Priority road: |0.48
|End of no passing by vehicles over 3.5 metric tons: |0.30
|No passing for vehicles over 3.5 metric tons: |0.19
|End of no passing: |0.01
|Roundabout mandatory: |0.01
