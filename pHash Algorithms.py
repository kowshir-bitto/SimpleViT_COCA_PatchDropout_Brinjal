import os
import cv2
import numpy as np
from collections import defaultdict

image_directory = 'D:/Brinjal/new Brinjal/Wet rot'

def dhash(image, hash_size=8):
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

hash_dict = defaultdict(list)

for image_name in os.listdir(image_directory):
    if image_name.endswith(('png', 'jpg', 'jpeg')):
        image_path = os.path.join(image_directory, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            hash_value = dhash(image)
            hash_dict[hash_value].append(image_path)

for hash_value, image_paths in hash_dict.items():
    if len(image_paths) > 1:
        for image_path in image_paths[1:]:
            os.remove(image_path)
            print(f"Deleted duplicate image: {image_path}")

print("Duplicate images deletion complete.")