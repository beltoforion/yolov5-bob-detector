"Deleting existing training images"
Remove-Item -Path ./bob-detector/images/train/*.jpg
Remove-Item -Path ./bob-detector/images/test/*.jpg

"Deleting existing training labels"
Remove-Item -Path ./bob-detector/labels/train/*.txt
Remove-Item -Path ./bob-detector/labels/test/*.txt
Remove-Item -Path ./bob-detector/labels/train.cache

"Copying new Files and labels"
python ./copy_and_augment_images.py -f ./bob-detector
