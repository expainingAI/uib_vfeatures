# UIB - V Features

## Introduction

Developing a solution in Shallow Learning needs the search and use of hardcoded features.

*UIB - V Features* provide a set of useful features. With three types of features: morphological, textures 
and color. All the features can be used with mask of with the contours.
 
The morphological features are all grouped in one iterator, so you can calculate all the features inside 
a loop easily.

---

## Demo

We're goin to use our library with a mask image.

```python
from uib_vfeatures.masks import Masks
from uib_vfeatures import Features_mask as ftrs
import cv2

```
First of all we read the image from a file, then we try our features with visualizations. All of the three features that
we use had a visual representation of it.

```python
mask = cv2.imread("mask.jpg")

Masks.bounding_box_area(mask, True)

Masks.eccentricity(mask, True)
Masks.solidity(mask, True)
```

###Iterator

You can use an iterator and implement every morpholical feature. 

```python
features = {}

for key, func in features.items():
    features[key] = func(mask)

```
Now we had a dicctionary of the form *{'Feature_name': value}*
