# ResUNet++ model in Keras TensorFlow

## Requirements:
	os
	numpy
	cv2
	tensorflow
	glob
	tqdm

## Folders:
	data: Contains the set of three dataset as mentioned.
	files: Contains the csv file and weight file generated during training.
	new_data: Contains two subfolder `images` and `masks`, they contains the augmented images and masks.

## Files:
	1. process_image.py: Augment the images and mask for the training dataset.
	2. data_generator.py: Dataset generator for the keras.
	3. infer.py: Run your model on test dataset and all the result are saved in the result` folder. The images are in the sequence: Image,Ground Truth Mask, Predicted Mask.
	4. run.py: Train the unet.
	5. unet.py: Contains the code for building the Unet architecture.

First check for the correct path and the patameters.
1.	$ python3 process_image.py - to augment training dataset.
2.	$ python3 run.py - to train the model.
3.	$ python3 infer.py - to test and generate the mask.
