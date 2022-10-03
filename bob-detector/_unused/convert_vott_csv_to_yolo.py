import csv
import cv2
import os
import numpy as np


def read_csv():
    with open('./vott-csv-export/Bob-Detector-export.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        line_count = 0

        dict = {}

        for row in reader:
            if line_count == 0:
                line_count += 1
                continue

            line_count += 1
            image_file, xmin, ymin, xmax, ymax, label = row
            box_def = [image_file, float(xmin), float(ymin), float(xmax), float(ymax), label]

            if box_def[0] in dict.keys():
                dict[box_def[0]].append(box_def)
            else:
                dict[box_def[0]] = []
                dict[box_def[0]].append(box_def)

        return dict


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def create_yolo_annotations(dict):
    for file_name in dict:
        file_path = f'./vott-csv-export/{file_name}'

        image = cv2.imread(file_path)
        h, w, c = image.shape
        print(f'{file_name}: {w} x {h}')

        annotation_file_path, ext = os.path.splitext(file_path)
        annotation_file_path = annotation_file_path + '.txt'

        with open(annotation_file_path, 'w+') as f:
            box_data = dict[file_name]
            for i in range(0, len(box_data)):
                xmin, ymin, xmax, ymax = box_data[i][1:5]

                xc = (xmin + (xmax - xmin)/2) / w
                yc = (ymin + (ymax - ymin)/2) / h

                wr = (xmax-xmin) / w
                hr = (ymax-ymin) / h            

                f.write(f'0 {xc} {yc} {wr} {hr}\n')            
                
        #
        # Create augmentation images by rotating to original image
        #

        for i in range(1, 4):
 #           image_center = tuple(np.array(image.shape[1::-1]) / 2)            
 #           mat_rot = cv2.getRotationMatrix2D(image_center, i * 90, 1.0)
            img_rot = rotate_image(image, i*90)
            
#            cv2.warpAffine(image, mat_rot, image.shape[1::-1], flags=cv2.INTER_LINEAR)

            anot_img_file_path, ext = os.path.splitext(file_path)
            anot_img_file_path = anot_img_file_path + f'_aug_{i}.jpg'
            cv2.imwrite(anot_img_file_path, img_rot)
            
            anot_txt_file_path, ext = os.path.splitext(anot_img_file_path)
            anot_txt_file_path = anot_txt_file_path + f'.txt'

            with open(anot_txt_file_path, 'w+') as f:
                box_data = dict[file_name]
                for j in range(0, len(box_data)):
                    xmin, ymin, xmax, ymax = box_data[j][1:5]

                    if i==1:
                        yc = 1-(xmin + (xmax - xmin)/2) / w
                        xc = (ymin + (ymax - ymin)/2) / h
                        hr = (xmax-xmin) / w
                        wr = (ymax-ymin) / h            
                    elif i==2:
                        xc = 1-(xmin + (xmax - xmin)/2) / w
                        yc = 1-(ymin + (ymax - ymin)/2) / h
                        wr = (xmax-xmin) / w
                        hr = (ymax-ymin) / h            
                    elif i==3:
                        yc = (xmin + (xmax - xmin)/2) / w
                        xc = 1-(ymin + (ymax - ymin)/2) / h
                        hr = (xmax-xmin) / w
                        wr = (ymax-ymin) / h            

                    f.write(f'0 {xc} {yc} {wr} {hr}\n')  

def main():
    dict = read_csv()

    for file_name in dict:
        print(f'\n{file_name}')

        box_data = dict[file_name]
        for i in range(0, len(box_data)):
            print(f'    {box_data[i][1]}, {box_data[i][2]}, {box_data[i][3]}, {box_data[i][4]}, {box_data[i][5]}')

    create_yolo_annotations(dict)

main()

