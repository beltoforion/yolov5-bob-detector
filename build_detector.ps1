"Deleting existing training images"
Remove-Item -Path ./bob-detector/images/train/*.jpg
Remove-Item -Path ./bob-detector/images/test/*.jpg

"Deleting existing training labels"
Remove-Item -Path ./bob-detector/labels/train/*.txt
Remove-Item -Path ./bob-detector/labels/test/*.txt
Remove-Item -Path ./bob-detector/labels/train.cache

"Copying new Files and labels"
python ./copy_and_augment_images.py -f ./bob-detector

"Training object detector"
Remove-Item -Path ../yolov5/runs/train/bob-detector -Force -Recurse -erroraction 'silentlycontinue'
python ../yolov5/train.py --img 640 --epochs 4 --data bob-detector.yaml --weights yolov5s.pt --name 'bob-detector'

"Testing detector"
Remove-Item -Path ../yolov5/runs/detect/bob-detector -Force -Recurse -erroraction 'silentlycontinue'
python ../yolov5/detect.py --name 'bob-detector' --source ./bob-detector/images/test/ --weights ../yolov5/runs/train/bob-detector/weights/best.pt