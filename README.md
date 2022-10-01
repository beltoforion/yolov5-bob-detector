# Probe Tip Training Images

This repository contains the YOLOv5 training data for the SENTIO probe tip detector. You will also need to check out YOLOv5
in order to train new images.

## Folder structure:

<pre>
sentio-tip-training
  |
  |--+ _annotated_originals
  |  +-- _tip_bottom
  |  +-- pyramide  
  |  +-- pyramide_spot_light
  |  +-- tip_bottom_classified_as_regular_tip  
  |  +-- tip_top 
  |  +-- tip_top_blurry   
  |  +-- titan
  |  +-- vpc 
  |--+ _labelimg        <- This folder contains the image labelling tool
     +-- data
        |-- predefined_classes.txt   <- Tip class definitionf for the labelling tool
  |--+ images                 
  |  +-- test           <- images for testing (optional)
  |  +-- train          <- training images folder; MUST BE EMPTY! the "augment_images.py" script will copy images into this folder
  +-- labels
  |  +-- train          <- training image labels folder; MUST BE EMPTY! the "augment_images.py" script will copy images into this folder  
  + augment_images.py
</pre>

YOLOv5 will take the training images from the "images/train" folder and the labels from the "labels/train" folder. The content of this folders is automatically created by executing the script "augment_images.py". DO NOT CHANGE the content of "images/train" or "labels/train" directly!

Steps to create copy the proper training images:
* delete every image from "images/train"
* create all text files in "images/labels"
* Open the powershell and type:

<pre>
<code>
python augment_images.py
</code>
</pre>

## How to create new training images

* start the labelling tool from the _labelimg subfolder. (You could also download it but inthen you have to manually set up the proper predefined classes)
* in the labelling tool select "Open Dir" and open any image folder in "_annotated_originals". You should see something like this:

![Unbenannt](https://user-images.githubusercontent.com/2202567/192758364-325d8fb6-ca3b-4606-a232-bf9d3d7d5867.PNG)

* If you do NOT see any training boxes or the images are not listed. Check that you have a file named "classes.txt" in the training folder. This file
  contains the SENTIO tip class definitions for "labelimg" without this file the software cannot load the files correctly!
* You can drop new images into the an image folder and properly annotate it by drawing boxes around the tips and assigning a tip type.
   * TAKE utmost care when doing the annotation! Bad training annotation can ruin an entire model! (Garbage in, Garbage out!)
   * Boxes must be consistent and as small as possible. Only draw the box around the feature that you want to end up in the model. For instance: For pyramide tips you cannot draw some boxes for the entire lengthy pyramide tip structure and some for the square shaped tip only. This would confuse the model!
   * Make sure the assigned tip type is correct. Currently you can assign any of the following types: 
        * "tip"    - class id: 0 cantilever , celadon or single prope tips seen from above or below
        * "unused" - class id: 1 unused, this class id was used for tips seen from below. This class was merged into the "tip" class.
        * "vpc"    - class id: 2 only for vertical probe tips
        * "pyr"    - class id: 3 pyramide probe tips (does not matter whether ring light or spot light)
        * "hf"     - class id: 4 any kind of HF probe. (Mostly titan)

## Purpose of the augmentation script

### What does the script do
* copy the annotaded images from the "_annotated_originals" subdirectories.
* copy the labels from the "_annotated_originals" subdirectories into the YOLOv5 label folder.
* for each image 4 training images will be created. Each one rotated by 90°. (Image augmentation)
* folders starting with "_" will be ignored!

Never ever make any change directly in the "labels/train" or "images/train" folders. The training requires a very high number of training images.
Therefore image augmentation is performed as a preprocessing step. Each training image is modified and this modification is also added to the training pool. 
Currently the only form of image augmentation is that a training image is rotated by multiples og 90°. We do not always have real images for probecards in all possible orientations so we simply create them be rotating the image. This will make detection results invariant to the probe card orientation (which can only change by multiples of 90°).

### Filtering and selecting training images
You can ignore image folders by prefixing them with an underscore ("_"). Thos folders will be skipped by the augmentation script. This is usefull for 
testing how certain training images will affect the final yolo model. For insstance you may want to add blurry tip images but they may actually decrease the quality of the model. If you drop those into the "tip_top" folder you cannot easily remove them later if the model does not perform well.




