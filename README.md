# OCT Noise Cancellation
## Developed by John Vu
## Latest Update: 04-16-2025

### Purpose
The purpose of this application developed is to take a bmp file of an OCT image and remove all unnecessary noise from the image, only leaving the outer layer and highlighting the outer layer in white, while leaving everything else in black background.

### Accuracy
The current model runs with a 99.45% accuracy of canceling out unnecessary noise on a 300 OCT image sample.

### Models
There are currently 3 models-- 1 model that is mainly for grayscaled backgrounds, 1 model that is for black backgrounds, and the last model is for when the eye is shaped undercurved.
