TRAIN_RATIO = 0.8

PROCESSED_PATH = r'D:\vscode_Projects\ArchaeoHack-Group-Still-Loading\archaeohack\processed_data'
DATA_PATH = r'D:\vscode_Projects\ArchaeoHack-Group-Still-Loading\archaeohack\data\me-sign-examples-pjb'

import cv2
import os
import random
import shutil
import numpy as np


def split_data(data_path):
    """List files in `data_path` and return the list.

    This function prints a short summary rather than dumping the whole list
    so output remains readable in terminals.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Path does not exist: {data_path}")

    
    # Skip hidden directories (starting with dot)
    print(f"Listing files in: {data_path}")
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.startswith('.'):
                continue

            filename = file
            parent_dir = os.path.basename(root)
            print(f"Found file: {filename} in directory: {parent_dir}")

            src_path = os.path.join(root, file)
            img = cv2.imread(src_path,cv2.IMREAD_GRAYSCALE)

            # If cv2.imread failed, img will be None
            if img is None:
                print(f"Warning: failed to read image: {src_path} (skipping)")
                continue
            
            # split into train and val sets
            random_num = random.random()
            print(f"RNGesus: {random_num}")
            if random_num < TRAIN_RATIO:
                subset = 'train'
            else:
                subset = 'val'
            
            #duplicate if train
            augimg = augment_pic(img)
            if subset == 'train':
                augimg2 = augment_pic(img)


            out_dir = os.path.join(PROCESSED_PATH, subset, parent_dir)
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                print(f"Error creating directory {out_dir}: {e}")
                continue

            out_path = os.path.join(out_dir, filename)
            try:
                ok = cv2.imwrite(out_path, augimg)
                if subset == 'train':
                    out_path2 = os.path.join(out_dir, 'dupe_' + filename)
                    ok = cv2.imwrite(out_path2, augimg2)
            except Exception as e:
                print(f"Exception when writing {out_path}: {e}")
                ok = False

            if not ok:
                print(f"Failed to write image: {out_path}")
            else:
                print(f"Wrote image: {out_path}")




    

    
    # Print a short summary so it's obvious something ran
    # print(f"Found {len(all_files)} files in: {data_path}")
    # print("First 10 entries:", all_files)
    # return all_files


def augment_pic(pic):
    # add random number of circles noise 5px diameter
    num_circles = random.randint(1, 30)
    for _ in range(num_circles):
        center_x = random.randint(0, pic.shape[1] - 1)
        center_y = random.randint(0, pic.shape[0] - 1)
        color = 0
        cv2.circle(pic, (center_x, center_y), 3, color, -1)
    # convert into binary
    _, pic = cv2.threshold(pic, 150, 255, cv2.THRESH_BINARY)
    # random erode/dilate value
    erodedilate = random.randint(-2, 2)
    if erodedilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erodedilate, erodedilate))
        transformed = cv2.dilate(pic, kernel, iterations=1)
    elif erodedilate < 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-erodedilate, -erodedilate))
        transformed = cv2.erode(pic, kernel, iterations=1)
    else:
        transformed = pic
    resized = cv2.resize(transformed, (200, 200))
    # random shift
    vertical_shift = random.randint(-40, 40)
    horizontal_shift = random.randint(-40, 40)
    M = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])
    pic = cv2.warpAffine(resized, M, (resized.shape[1], resized.shape[0]),borderValue=255)
    return pic


def main():
    print('splitting data...')
    # Use a raw string or os.path.join to avoid accidental escape sequences
    try:
        split_data(DATA_PATH)
    except Exception as e:
        print('Error while splitting data:', repr(e))
    else:
        print('data split complete.')


if __name__ == '__main__':
    main()