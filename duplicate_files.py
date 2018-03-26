import os
from shutil import copyfile

DATA_ROOT = os.path.expanduser("~/data/101_ObjectCategories_duplicated")

for subdir, dirs, files in os.walk(DATA_ROOT):
    for file in files:
        if os.path.splitext(file)[-1] == ".jpg": #filtering .DS_name and non image files
            copyfile(os.path.join(subdir, file), os.path.join(subdir, file.replace(".jpg","_copy.jpg")))
