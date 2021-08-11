#from imutils import paths
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import mean_squared_error
from math import sqrt

def dhash(image, hashSize=12):
    """
    convert the image to grayscale and resize the grayscale image,
    adding a single column (width) so we can compute the horizontal gradient.
    compute the (relative) horizontal gradient between adjacent column pixels.
    convert the difference image to a hash and return it

    Args:
        image:
        hashSize:

    Returns:

    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def find_dupl(path: str, trashold: int) -> dict():
    """
    Search dublicate images in directory
    Args:
        path: path with images
        trashold: rms

    Returns:
    Dictionary, where key is a base image and value is array with dublicates (may be delete)
    """
    to_del = {}
    images = [img for img in os.listdir(path) if img.endswith('jpg')]
    images.sort()
    im1 = np.array(cv2.imread(path + '/' + images[0], cv2.IMREAD_GRAYSCALE))
    key_name = images[0]
    rms = 0
    for n in range(len(images)-1):
        im2 = np.array(cv2.imread(path + '/' + images[n+1], cv2.IMREAD_GRAYSCALE))
        rms = sqrt(mean_squared_error(im1, im2))
        if rms < trashold:
            p = to_del.get(key_name, [])
            p.append(images[n+1])
            to_del[key_name] = p
        else:
            im1 = im2.copy()
            key_name = images[n+1]
            # print(key_name)
    return to_del


def move_dublicates(path: str, trashold: int):
    '''
    Rename and place dublicates to "dublicates"

    Args:
        path: path with images
        trashold: default 3 (rms)

    Returns:

    '''

    dublicates = find_dupl(path, trashold)

    try:
        os.mkdir(os.path.join(path, 'dublicates'))
    except:
        pass

    for (img, dublicates) in dublicates.items():
        dublicates.sort()
        for dublicate in dublicates:
            shutil.move(os.path.join(path, dublicate), os.path.join(path, 'dublicates', str(img) + '_' + dublicate))
            # print(str(img) + '_' + dublicate)

def restore_dublicates(path: str):
    """
    Rename and replace files from 'dublicates'

    Args:
        path: path with images

    Returns:

    """
    old_path = os.path.join(path, 'dublicates')
    files = os.listdir(old_path)
    for p in files:
        if p.endswith('jpg'):
            try:
                shutil.move(os.path.join(old_path, p), os.path.join(path, p.split('jpg_')[1]))
            except:
                pass
    if len(os.listdir(old_path)) == 0:
        shutil.rmtree(old_path)

def packer_by_cnt(path, pack_cnt=1000):
    n = 0
    files = os.listdir(path)
    files.sort()
    dir_num = 0
    for f in files:
        dir_num = n // pack_cnt
        if n % pack_cnt == 0:
            try:
                os.mkdir(os.path.join(path, str(dir_num).zfill(2) + '_pack'))
            except:
                pass

        if f.endswith('jpg'):
            # print(dir + '/' + f, dir + '/' + dir_num + '_pack')
            shutil.move(os.path.join(path, f), os.path.join(path, str(dir_num).zfill(2) + '_pack', f))
            n += 1

def packer_by_cnt(path, pack_cnt=1000):
    n = 0
    files = os.listdir(path)
    files.sort()
    for f in files:
        dir_num = n // pack_cnt
        if n % pack_cnt == 0:
            os.makedirs(os.path.join(path, str(dir_num).zfill(2) + '_pack'), exist_ok=True)

        if f.endswith('jpg'):
            # print(dir + '/' + f, dir + '/' + dir_num + '_pack')
            shutil.move(os.path.join(path, f), os.path.join(path, str(dir_num).zfill(2) + '_pack', f))
            n += 1

def packer_by_size(path, pack_size=10.0):
    n = pack_size * 1024**2
    files = os.listdir(path)
    files.sort()
    dir_num = -1
    for f in files:
        if n >= pack_size * 1024**2:
            dir_num += 1
            n = 0
            os.makedirs(os.path.join(path, str(dir_num).zfill(2) + '_pack'), exist_ok=True)


        if f.endswith('.jpg'):
            # print(dir + '/' + f, dir + '/' + dir_num + '_pack')
            n += os.path.getsize(os.path.join(path, f))
            shutil.move(os.path.join(path, f), os.path.join(path, str(dir_num).zfill(2) + '_pack', f))


# поиск в подпапках
if False:
    for path in os.listdir(dpath):
        imagePaths = os.path.join(dpath, path)
        if os.path.isdir(imagePaths):
            print(imagePaths)

            move_dublicates(imagePaths)

# loop over our image paths
# for imagePath in os.listdir(imagePaths):
# 	# load the input image and compute the hash
# 	if imagePath.endswith('jpg'):
# 		image = cv2.imread(imagePaths + '/' + imagePath)
# 		h = dhash(image)
# 		# grab all image paths with that hash, add the current image
# 		# path to it, and store the list back in the hashes dictionary
# 		p = hashes.get(h, [])
# 		p.append(imagePath)
# 		hashes[h] = p
#




dpath = r'/home/drova326/Загрузки/kic_vlad/smena'
trashold = 4.0
dublicates = find_dupl(dpath, trashold)

#подбор чувствительности
if False:
    for img, dublicates in dublicates.items():
        # check to see if there is more than one image with the same hash
        if len(dublicates) > 0:
            # initialize a montage to store all images with the same hash
            montage = cv2.imread(os.path.join(dpath, img))
            montage = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)
            montage = cv2.resize(montage, (300, 300))
            montage = cv2.putText(montage, 'Original', (110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            # loop over all image paths with the same hash
            for dublicate in dublicates:
                image = cv2.imread(os.path.join(dpath, dublicate))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (300, 300))
                montage = np.hstack([montage, image])
            print("[INFO] hash: {}".format(img))
            # cv2.imshow("Montage", montage)
            # plt.imshow(img, cmap='gray')
            plt.imshow(montage)
            plt.show()
else:
    if False:
        move_dublicates(dpath, trashold)

    if False:
        restore_dublicates(dpath)


packer_by_size(dpath, pack_size=10.0)

print(1)

