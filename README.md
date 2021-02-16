<div align="center">
  <img src="https://raw.githubusercontent.com/explainingAI/uib_vfeatures/master/docs/uib-vfeatures.png">
</div>


# UIB - V Features


![PyPI - Python Version](https://img.shields.io/pypi/pyversions/uib-vfeatures)
[![PyPI version](https://badge.fury.io/py/uib-vfeatures.svg)](https://badge.fury.io/py/uib-vfeatures)



**UIB - V Features** is a library to extract useful features of different types(morphological, texture 
and color). To increase the flexibility of the library all functions can be used with masks or contours. 

UIB - V Features was originally developed by researchers of [UGIVIA](http://ugivia.uib.es/). This research group
is centered on computer vision and artificial intelligence. 

All the features available are grouped in one iterator to simplify the use of this library.

### Installation

Install the library is very simple with pip

```
$ pip install uib-vfeatures
```

---
## List of features

### Morphological

*   Solidity
*   Convex hull perimeter
*   Convex hull area
*   Bounding box area
*   Rectangularity
*   Minor radius
*   Maximum radius
*   Feret
*   Breadh
*   Circularity
*   Roundness
*   Feret Angle
*   Eccenctricity
*   Center
*   Sphericity
*   Aspect Ratio
*   Area equivalent diameter
*   Perimeter equivalent diameter
*   Equivalent elipse area
*   Compactness
*   Area
*   Convexity
*   Shape
*   Perimeter

### Color

*   Mean of the LAB channels
*   Mean of the RGB channels
*   Mean of the HSV channels
*   Standard deviation of the LAB channels
*   Standard deviation of the RGB channels
*   Standard deviation of the HSV channels


#### Texture features

+   Contrast
+   Dissimilarity
+   Homogeneity
+   ASN
+   Energy
+   Correlation

---

## Demo

We're going to use our library with a mask image .

```python
from uib_vfeatures.masks import Masks
from uib_vfeatures import Features_mask as ftrs
import cv2

```
First of all we read the image from a file, then we try our features with visualizations. We only have 
three features with visualization: the bounding box area, the eccentricity and the solidity. 

```python
mask = cv2.imread("mask.jpg")

Masks.bounding_box_area(mask, True)

Masks.eccentricity(mask, True)
Masks.solidity(mask, True)
```

### Iterator

You can use an iterator and implement every morpholical feature. 

```python
features = {}

for key, func in features.items():
    features[key] = func(mask)

```
As a result we had a dicctionary of the form *{'Feature_name': value}*

### Citation

If you use this code, please cite

```
@article{PETROVIC2020104027,
    title = {Sickle-cell disease diagnosis support selecting the most appropriate machine learning method: Towards a general and interpretable approach for cell morphology analysis from microscopy images},
    author = {Nataša Petrović and Gabriel Moyà-Alcover and Antoni Jaume-i-Capó and Manuel González-Hidalgo},
    journal = {Computers in Biology and Medicine},
    volume = {126},
    pages = {104027},
    year = {2020},
    issn = {0010-4825},
    doi = {https://doi.org/10.1016/j.compbiomed.2020.104027},
    url = {https://www.sciencedirect.com/science/article/pii/S0010482520303589}
}

```