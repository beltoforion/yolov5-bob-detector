import cv2
import os
import pathlib
import hashlib
import numpy as np
import csv
import shutil
import argparse

images_written : int = 0

def get_md5_hash(img_path):
    with open(img_path, "rb") as f:
        img_hash = hashlib.md5()
        while chunk := f.read(8192):
           img_hash.update(chunk)

    return img_hash.hexdigest()


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

def copy_annotation_file(annot_file, outfile, angle, image):
    h, w, c = image.shape

    # read the annotation file
    with open(str(annot_file), 'r', newline='') as csv_in_file:
        with open(str(outfile), 'w+', newline='') as csv_out_file:
            reader = csv.reader(csv_in_file, delimiter=' ')
        
            for row in reader:
                classid = int(row[0])
                if classid>=4:
                    print(">>>>>  BAD CLASS ID  <<<<")
                    
                xo = float(row[1])
                yo = float(row[2])
                wo = float(row[3])
                ho = float(row[4])

                if angle==0:
                    csv_out_file.write(f'{classid} {xo} {yo} {wo} {ho}\n')
                elif angle==90:
                    csv_out_file.write(f'{classid} {yo} {1-xo} {ho} {wo}\n')
                elif angle==180:
                    csv_out_file.write(f'{classid} {1-xo} {1-yo} {wo} {ho}\n')
                elif angle==270:
                    csv_out_file.write(f'{classid} {1-yo} {xo} {ho} {wo}\n')


def copy_and_augment_file(file, target_folder : pathlib.Path):
    global images_written

    md5 = get_md5_hash(str(file)) 
    outfile = target_folder / 'images' / 'train' / md5

    img = cv2.imread(str(file))

    progress = ''

    for i in range(4):
        angle = i * 90
        filename = md5 + f'_rot_{angle}'
        outfile = target_folder / 'images' / 'train' / filename
        outfile = outfile.with_suffix('.jpg')

        # copy file
        if i!=0:
            img = np.rot90(img)

        cv2.imwrite(str(outfile), img)
        images_written += 1
        progress = progress + "."

        # copy annotations
        outfile = target_folder / 'labels' / 'train' / filename
        outfile = outfile.with_suffix('.txt')
        copy_annotation_file(file.with_suffix('.txt'), outfile, angle, img)

    print(f" - {images_written}: {progress}")


def copy_file(file, target_image_folder : pathlib.Path, target_annot_folder : pathlib.Path = None):
    md5 = get_md5_hash(str(file)) 
    outfile = target_image_folder / f'{md5}{file.suffix}'

    print(f"        Copying file: {outfile}")
    shutil.copyfile(str(file), outfile)

    if not target_annot_folder is None:
        annotation_file = file.with_suffix('.txt')
        if annotation_file.exists():
            print(f"        Copying annotation-file: {annotation_file}")
            shutil.copyfile(str(annotation_file), target_annot_folder / f'{outfile.stem}.txt')


def copy_background_images(source_folder : pathlib.Path, target_folder : pathlib.Path):
    global script_dir
    print(f'scanning background image folder {source_folder}:')
    for file in source_folder.iterdir():
        # iterate over all jpegs
        if file.suffix.lower() != '.jpg' and file.suffix.lower() != '.jpeg':
            continue

        print(f'copying background image {file}:')
        copy_file(file, target_folder / 'images' / 'train')


def copy_test_images(source_folder : pathlib.Path, target_folder : pathlib.Path):
    global script_dir
    print(f'scanning test image folder {source_folder}:')
    for file in source_folder.iterdir():
        # iterate over all jpegs
        if file.suffix.lower() != '.jpg' and file.suffix.lower() != '.jpeg':
            continue

        print(f'copying test image {file}:')
        copy_file(file, target_folder / 'images' / 'test', target_folder / 'labels' / 'test')


def process_folder(folder, target_folder : pathlib.Path):
    global script_dir

    # process folder.
    # The folder must contain annotated images. Therefore for each image an identically named txt 
    # file with annotations has to exists. If not the image is ignored.
    print(f'scanning {folder}:')
    for file in folder.iterdir():
        # iterate over all jpegs
        if file.suffix != '.jpg' and file.suffix != '.JPG' and file.suffix != '.jpeg':
            if file.suffix.lower() == '.png' or file.suffix.lower() == '.bmp':
                raise Exception(f'PNG and BMP files cannot be processed by this script! (file: {file.resolve()})');
                
            continue

        # check if annotation file exists:
        annotation_file = file.with_suffix('.txt')
        if not annotation_file.exists():
            raise Exception(f'Annotation file is missing! Only images in the background folder can omit the annotation file! (missing file is "{annotation_file.resolve()}")')
        else:
            print(f'    Augmenting {file.name}', end='')

        # Great we have an image file and an annotation file. Now copy them both to the training folder
        # but we will make the md5 checksum the new file name.
        copy_and_augment_file(file, target_folder)


def dir_path(dir_path : str):
    if os.path.isdir(dir_path):
        return dir_path
    else:
        raise NotADirectoryError(dir_path)


def main():
    parser = argparse.ArgumentParser(description='Copy and Augment Training images')
    parser.add_argument("-f", "--Folder", dest="target_folder", help='The folder to run this script on', required=True, type=dir_path)

    args = parser.parse_args()

    print('\r\n')
    print('Copy and Augment Training images')
    print('----------------------------------')
    print(f' - target folder: {args.target_folder}')

    if (not os.path.exists(args.target_folder)):
        print(f'The target folder {args.target_folder} does not exists')
        exit(-1)

    annotated_originals_dir = pathlib.Path(args.target_folder) / '_annotated_originals'
    if not annotated_originals_dir.exists():
        raise Exception(f'Annotation file folder \'{annotated_originals_dir}\' not found!')
    
    for dirpath, dirs, files in os.walk(annotated_originals_dir):
        for dir in dirs:
            folder = pathlib.Path(dirpath) / dir
            if dir.startswith('_'):
                print(f'skipping folder \'{dir}\'')
                continue
            elif 'background' in dir:
                print(f'copying background images from \'{dir}\':')
                copy_background_images(folder, pathlib.Path(args.target_folder))
                continue
            elif '_test' in dir:
                print(f'copying test images from \'{dir}\':')
                copy_test_images(folder, pathlib.Path(args.target_folder))                
                continue
            else:
                print(f'processing folder \'{dir}\':')
                process_folder(folder, pathlib.Path(args.target_folder))
            

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print('\r\nError:')
        print('------')
        print(f'{str(exc)}\r\n')
