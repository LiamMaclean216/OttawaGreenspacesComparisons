# OttawaGreenComparisons
========================
All the code in this repository is the result of my summer internship at LAGISS in 2019. The objective of this internship was to use deep learning to evaluate the quality of the greenspaces of Ottawa from street view images. The process of gathering image comparisons to feed a siamese model can be used to evaluate many different characteristics including the quality of greenspaces or urban walkability. So this code was developped in order to create the image dataset that must be feed to this kind of neural network. It includes:
	- a web interface in php which stores comparisons results in a SQL database
	![Interface](/Documentation/images/web_interface.PNG)
	- pythons classes for image processing 
	![Histogramme](/Documentation/images/Histogram.jpg)
	- pythons functions to train and manipulate keras model
	![Data augmentation](/Documentation/images/augmented.png)



# Example of creation of images from a panorama
```
from Class_Image import Panorama
from os.path import abspath

SAVE_DIR = abspath('Examples/Test_images')
PANO = abspath("Examples/Test_images/-75.497094_45.457925__m3YDpWSlFIhCXx922pZLw_999_2019-03-08T13-06-51.jpg")

pano = Panorama(PANO, build_array=True)
pano.create_4_images(SAVE_DIR)
pano.plot_created_images("Examples/Test_images/4.jpg")
```
![Image creation](/Documentation/images/4_from_panos.png)

All the simple image use cases are described in the example.py file.

# Use git lfs to clone the repository (https://git-lfs.github.com/)

## Contact
* e-mail: guillaume.guebin@hotmail.fr
