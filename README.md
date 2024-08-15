# Deep SORT

## Introduction

This repository contains code for Object Tracking Algorithm based on SOTA deepSORT and YOLOv8.

## Dependencies

The code is compatible with Python 2.7 and 3. The following dependencies are
needed to run the tracker:

```
pip install -r requirements.txt
```
Additionally, feature generation requires TensorFlow (>= 1.0).

## Installation

First, clone the repository:
```
https://github.com/mazhar18941/deepSort-Yolov8.git
```
Then, download the CNN checkpoint file from
[here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp).



We have replaced the appearance descriptor with a custom deep convolutional
neural network (see below).

## Running the tracker

```
python app.py --descriptor "path to descriptor" --object-detector "path to yolov8" --video "path to video"
```
Only "car" class is being tracked in code. In order to track other class like "person" change code lne no 44 in app.py to following:
if result.names[box.cls[0].item()] == 'person':
Check `python app.py -h` for an overview of available options.


## Highlevel overview of source files


In package `deep_sort` is the main tracking code:

* `detection.py`: Detection base class.
* `kalman_filter.py`: A Kalman filter implementation and concrete
   parametrization for image space filtering.
* `linear_assignment.py`: This module contains code for min cost matching and
   the matching cascade.
* `iou_matching.py`: This module contains the IOU matching metric.
* `nn_matching.py`: A module for a nearest neighbor matching metric.
* `track.py`: The track class contains single-target track data such as Kalman
  state, number of hits, misses, hit streak, associated feature vectors, etc.
* `tracker.py`: This is the multi-target tracker class.

## References

https://github.com/nwojke/deep_sort
https://github.com/Qidian213/deep_sort_yolov3
