import os

dst = "../data/masks"

with open('masks.txt') as my_file:
    for filename in my_file:
        src = os.path.join("masks/", filename.strip() ) # .strip() to avoid un-wanted white spaces
        os.rename(src, os.path.join(dst, filename.strip().removesuffix("png")+"gif"))