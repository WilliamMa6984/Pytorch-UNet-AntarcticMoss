# Source - https://stackoverflow.com/a/45212865
# Posted by Shai, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-10, License - CC BY-SA 3.0

import os

dst = "data/imgs"

with open('dataset/train_img.txt') as my_file:
    for filename in my_file:
        src = os.path.join("dataset/train/", filename.strip() ) # .strip() to avoid un-wanted white spaces
        os.rename(src, os.path.join(dst, filename.strip()))

dst = "data/masks"

with open('dataset/train_mask.txt') as my_file:
    for filename in my_file:
        src = os.path.join("dataset/train/", filename.strip() ) # .strip() to avoid un-wanted white spaces
        os.rename(src, os.path.join(dst, filename.strip()))





dst = "test_data/imgs"

with open('dataset/valid_img.txt') as my_file:
    for filename in my_file:
        src = os.path.join("dataset/test/", filename.strip() ) # .strip() to avoid un-wanted white spaces
        os.rename(src, os.path.join(dst, filename.strip()))

dst = "test_data/masks"

with open('dataset/valid_mask.txt') as my_file:
    for filename in my_file:
        src = os.path.join("dataset/test/", filename.strip() ) # .strip() to avoid un-wanted white spaces
        os.rename(src, os.path.join(dst, filename.strip()))
