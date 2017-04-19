# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[balanced data]: ./comma.ai/comma.ai_balanced_hist.png "Balanced Data Histogram"
[original data]: ./comma.ai/comma.ai_orig_hist.png "Original Data Histogram"
[comma.ai training]: ./comma.ai/comma.ai_training.png "Training MSE Loss"
[comma.ai model]: comma.ai_model.png "Modified Comma.ai model"
[input image 1]: ./examples/input1.png "Input image 1"
[input image 2]: ./examples/input2.png "Input image 2"
[processed image 1]: ./examples/processed1.png "Processes image 1"
[processed image 2]: ./examples/processed2.png "Processes image 2"


## Rubric Points
### Files Submitted & Code Quality

#####1. Submission includes all required files and can be used to run the simulator in autonomous mode

All project files are available at [Barney-s CarnND Repo](https://github.com/barney-s/CarND-Behavioral-Cloning-P3)

* model.py - Code to create and train the model
* drive.py - Using the model to drive the car in autonomous mode
* comma.ai/comma.ai_model.h5 - Trained CNN
* comma.ai/comma.ai_model.json - CNN Arch
* comma.ai/comma.ai_model.png - CNN Visualization
* comma.ai/data/balanced/ - Data used for training the CNN
* image.py - Camera image preprocessing
* video.py - convert autonomous driving image data to video
* writeup.md - Write up for the project

##### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car was driven autonomously around the track by executing the drive.py

```sh
python drive.py comma.ai 
```

##### 3. Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline that is used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

##### 1. An appropriate model architecture has been employed

The modified comma.ai model consists of

- CNN with 8x8, 5x5 filter sizes and depths between 16 and 64 (model.py: comma_ai_model).
- The model includes ```tanh``` layers to introduce nonlinearity.
- The images are cropped and normalized as part of preprocessing.

Other models tested include Nvidia model.

##### 2. Attempts to reduce overfitting in the model
- Dropout layers were added in order to reduce overfitting.
- The dataset was balanced to remove bias towards lowere steering angles from the input images.
- Curation of images with bad steering angles was done to help in acute curves.


##### 3. Model parameter tuning
- The model used an adam optimizer, so the learning rate was not tuned manually.
- The dropouts between different layers were tuned and tested.

##### 4. Appropriate training data
- Training data seemed to have the biggest impact on the performance in autonomous mode.
- Used udacity's training data augmented by training data from Eric L and my own training data for curves to keep the vehicle driving on the road.
- I used a combination of center lane driving and cornering data.



## References

1. https://arxiv.org/abs/1511.07289
2. https://arxiv.org/pdf/1511.07289.pdf
3. https://www.reddit.com/r/MachineLearning/comments/2x0bq8/some_questions_regarding_batch_normalization/?su=ynbwk&st=iprg6e3w&sh=88bcbe40
4. https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf