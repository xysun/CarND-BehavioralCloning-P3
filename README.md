## Project 3 documentation ##
This markdown file documents the details for my Project 3 submission.

### Training data

Training data is obtained through driving manually several laps with a PS4 joystick:

- One lap of driving that keeps to center of the road
- Simulate "recovery driving", by driving offcenter (without recording), and then record steering back to the center. This is done for all corners, for both drifting to left/right side.
- For small road segments that the model (will be discussed later) demonstrates difficulty, some more manual driving is done.

In total, 34k images (center, left, right camera) were recorded.

### Image augmentation

To generate more training data without actually driving, below augmentation methods were used:

- Flip each image and record opposite steering angles.
- Brightness adjust ([reference](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.q08muecvh))
- Use both left and right camera image, while adjust steering by a fixed delta (0.25)

After augmentation and using both left and right camera images, a total of 52k images were used.

Below are example images: 

Original image:

![Original image](img1.png)

Flipped:

![Flipped image](img1_flipped.png)

Brightness adjusted:

![Brightness adjusted](img1_brightness_adjusted.png)

Left camera:

![Left](img1_left.png)

Right camera:

![Right](img1_right.png)

### Data normalization

Every image was resized to 64x64, all the RGB values were normalized to range of [-0.5, 0.5]. 

### Training, validation split

To avoid overfitting, all data were split into training/validation using scikit-learn's `train_test_split` method. `test_size` was set to 25% of all data.

### Model description

I used same model from [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py) here.

This is a model with 3 convolution layers followed by 1 fully connected layer. There's a dropout layer after both the last convolution layer and the fully connected layer. Exponential Linear Units (ELU) was used as activation layer. Adam optimizer was used with mean squared error (mse) as objective.

Below are the dimensions:

- Conv1: filters = 16, kernel size = 8x8, strides = 4x4
- Conv2: filters = 32, kernel size = 5x5, strides = 2x2
- Conv2: filters = 64, kernel size = 5x5, strides = 2x2
- Dropout rate after 3 convolution layers: 0.2
- Fully connected layer: 512
- Dropout rate after the fully connected layer: 0.5

### Training

The model was trained using batch size 32 and in 10 epochs. 


### Performance

The model performed well on track 1, kept within the drive-able potion of road. 

I also tried running the model on track 2, it behaved well there too. The only thing I noticed is since track 2 has slopes, in rare occasions the car may not have enough initial speed to climb a steep uphill. The model kept the car in the right direction though.

### Further work

The model did not use any other driving data (throttle, whether it's breaking, speed). It would be interesting to explore how we can incorporate these additional features into self driving. 