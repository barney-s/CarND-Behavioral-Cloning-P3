# Behavioral Cloning Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Behavioral Cloning Project**

In this project, I have used what has been learned about deep neural networks and convolutional neural networks to clone driving behavior. A model was designed in Keras and trained. The model will output a steering angle to an autonomous vehicle.

A simulator is provided, where the car was steered around a track for data collection. The image data and steering angles collected were used to train a neural network and then use this model to drive the car autonomously around the track.


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
[comma.ai model]: ./comma.ai/comma.ai_model.png "Modified Comma.ai model"
[input image 1]: ./examples/input1.png "Input image 1"
[input image 2]: ./examples/input2.png "Input image 2"
[processed image 1]: ./examples/processed1.png "Processes image 1"
[processed image 2]: ./examples/processed2.png "Processes image 2"


## Rubric Points
### Files Submitted & Code Quality

##### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

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

### Model Architecture and Training

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

### Detailed Model Architecture and Training Strategy 
#### 1. Solution Design Approach

**Starting point**  
At the begining, i was using my recorded training data with a _Trivial Model_ and checking the loss. The car was behaving erraticaly. 

- Trivial model
- Captured training data from simulator
- No drouputs, regularization
- Normalized image in keras using Lambda layer
- augmentation - mirroring of images
- used data from all cameras with steering angle corection (0.2)

**Check different models stage**  
Tried other well known models: _Nvidia_, _comma.ai_ and both perfomed poorly. This stage was the most excrutiating in terms of understanding what was happening.

- few runs with trivial and comma.ai, nvidia models (elu activation)
- output was a single angle with minimal variations. failed to cross even the first curve.
- Cropped the image

**More Data**  
Suspecting that the input data wasn't good, tried adding more training data: Udacity training data, more lap training data. But the model still performed poorly.

**Adjusting Preprocessing**  
Tried different preprocessing techniques to understand their impact on the model performance.  
*Cropping*  
- More Cropping: This improved the performance  

*Image Adjustment*  
- CLAHE: This did not help much.
- Simple conversion to HSV colorspace helped.

*Normalization*  
- Normalized the image data (did not test without it - TODO ?)
    
With these the model performed better.

![input image 1] ![processed image 1] 
![input image 2] ![processed image 2] 


**Changing the activation function**  
The biggest improvement in performance came when i switched from ELU to RELU/Tanh. Finally settled on tanh since that seemed to have a better performance. (TODO quantify?)

**Balancing Data**  
The input data was skewed towards low steering angles. Since i could not find a balancing library that handled regression data, i implemented a trivial one (model.py: balance_steering_data) that selects the extreme steering angles (abs(angle) > 0.5) as is and samples the smaller angles with a probablity equal to abs(angle). This improved the model performace and helped with handling the curves better. The car was not negotiating the first few curves.

![original data]
![balanced data]

This also meant the training was happeneing on a lot fewer images, yet the model was performing better.

**Dropouts and Regularization**  
Over the several iterations, added droputs and data regularization to increase the distance driven succesfully. It also heled negotiating normal curves.

**Curation of training data**  
The model was still not negotiating certain acute curves. On slack channels came across an approach succesfully implemented by Eric Levine. He used carefully curated set of very few images to successfully train the CNN.  That inspired me to curated out (removing) negatively impacting training angles in the curves. This helped the model negotiate those curves.

**No Augmentation**  
Over several iterations of adjusting model parameters, preprocessing, balancing etc, i also turned off image augmentation to reduce the amount of training data. That seemed to marginally improve the performance. (TODO - quantify)


#### 2. Final Model Architecture

Here is a visualization of the modified comma.ai model that i used to succesfully complete a few laps on the lake track.

![comma.ai model]

The model consists of the following layers.  

```
Modified comma.ai model:
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 18, 80, 16)    3088        convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 18, 80, 16)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 9, 40, 32)     12832       dropout_1[0][0]                  
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 20, 64)     51264       convolution2d_2[0][0]            
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 5, 20, 64)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 6400)          0           dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 512)           3277312     flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 512)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 128)           65664       dropout_3[0][0]                  
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 128)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             129         dropout_4[0][0]                  
====================================================================================================
Total params: 3,410,289
Trainable params: 3,410,289
Non-trainable params: 0
```


#### 3. Creation of the Training Set & Training Process

As described in _1. Solution Design Approach_, I started off with a large training data with marginal success. Finaly once the data was balanced and curated to a smaller dataset, the model performed better. 

An example training loss plot:
![comma.ai training]

With very few training data, the training was much faster and even thought i used less epochs (and more loss), the model performed better.

Thought iam not able to intuitively quantify or explain, these were the observations:  

- small curated data + medium network - better performance
- small balanced data + medium network - better perfomance
- small data + large network (nvidia) - not great performance
- TODO: large data + large network 

## Usage
##### Environment  

- For Lab environment setup, click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md). 
- Simulator can be downloaded from the classroom.
- Sample training data is also provided if needed.

##### Training

Training data should have the run-data under a subdir.

```sh
├── sim_data
│   ├── eldata
│   │   ├── IMG/*.jpg
│   │   └── driving_log.csv
│   ├── udacity
│   │   ├── IMG/*.jpg
│   │   └── driving_log.csv
    ├── corners
    │   ├── IMG/*.jpg
    │   └── driving_log.csv
    └── run1
        ├── IMG/*.jpg
        └── driving_log.csv
        
```

Training is done using this command:

```sh
python model.py --model comma.ai  --datapath ~/Desktop/sim_data/ --epochs 20

# different model
python model.py --model nvidia  --drive --datapath ~/Desktop/sim_data/ --epochs 10

#To train and drive
python model.py --drive --model comma.ai  --datapath ~/Desktop/sim_data/ --epochs 20
```

By default the training balances the data and saves it under the model directory (model/data/balanced). It also created the model files (.json, .h5) as well as some visualization of the model, data balancing and training loss.

```sh
comma.ai
├── comma.ai.h5
├── comma.ai.json
├── comma.ai_balanced_hist.png
├── comma.ai_model.png
├── comma.ai_orig_hist.png
├── comma.ai_training.png
└── data
    └── balanced
        ├── IMG/*jpg
        └── comma.ai_input.csv

```

When using balanced data, we should skip balancing to avoid reduction of samples used in training:

```sh
python model.py --balanced --model comma.ai   --datapath comma.ai/data/ --epochs 20
```

##### Running in autonomous mode

```sh
# start simulator in autonomous mode and then

python drive.py comma.ai

# to save the run images
python drive.py comma.ai test_run

# to create a video of the run
python video.py test_run
```


## Video of a succesfull run
[Lake Track Video](https://github.com/barney-s/CarND-Behavioral-Cloning-P3/blob/master/video.mp4)

## Approaches not taken
- More image augmentation like rotation, tranformation, adjust brightness, add shadows, use more camera angles
- More training data - several runs, explicit recovery runs, explicit cornering segments
- Larger and wider model (nvidia)

## References

* https://arxiv.org/pdf/1511.07289.pdf
* https://www.reddit.com/r/MachineLearning/comments/2x0bq8/some_questions_regarding_batch_normalization/?su=ynbwk&st=iprg6e3w&sh=88bcbe40
* https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
* https://arxiv.org/abs/1610.02391v1
* slack channels