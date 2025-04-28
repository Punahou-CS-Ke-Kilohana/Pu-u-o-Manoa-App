r"""
DO NOT RUN UNLESS YOU KNOW WHAT THIS DOES!!!

Clears all saved test images. Used for debugging purposes only.
"""

import os
import shutil

confirm = (input("You are about to DELETE ALL SAVED TEST IMAGES. Confirm with 'y': ").lower() == 'y')
if confirm:
    # model root
    root = os.path.dirname(__file__)
    for item in os.listdir(root):
        item_path = os.path.join(root, item)

        if item == os.path.basename(__file__):
            # skip this file
            continue
        # remove items
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
    print("Deleted all saved test images.")
else:
    print("Kept all saved test images.")
