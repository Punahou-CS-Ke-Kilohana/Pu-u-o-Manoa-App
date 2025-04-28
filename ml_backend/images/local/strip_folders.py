#!/usr/bin/env python3
import os
import glob
import sys

def keep_first_image_only(directory):
    """
    Process each subfolder in the given directory and keep only the first image file 
    in each subfolder.
    """
    print(f"Starting to process subfolders in: {directory}")
    total_removed = 0
    
    # Get all subfolders
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    
    # Sort the subfolders for consistent processing
    subfolders.sort()
    
    for subfolder in subfolders:
        # Skip hidden folders (those starting with a dot)
        if os.path.basename(subfolder).startswith('.'):
            continue
            
        print(f"Processing subfolder: {subfolder}")
        
        # Define common image extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp', '*.tiff', '*.svg']
        
        # Collect all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(subfolder, ext)))
            # Case insensitive match for extensions (e.g., both .JPG and .jpg)
            image_files.extend(glob.glob(os.path.join(subfolder, ext.upper())))
        
        # Sort the files to ensure consistent behavior
        image_files.sort()
        
        if not image_files:
            print(f"  No image files found in {subfolder}")
            continue
        
        # Keep the first image
        first_image = image_files[0]
        images_to_delete = image_files[1:]
        
        print(f"  Keeping: {os.path.basename(first_image)}")
        
        # Delete all other images
        for img in images_to_delete:
            try:
                os.remove(img)
                print(f"  Deleted: {os.path.basename(img)}")
                total_removed += 1
            except Exception as e:
                print(f"  Error deleting {img}: {e}")
    
    print(f"\nSummary: Removed {total_removed} image files, keeping one image in each folder")

if __name__ == "__main__":
    # Use the current directory by default
    current_dir = os.getcwd()
    
    # Allow user to specify a different directory as a command-line argument
    if len(sys.argv) > 1:
        if os.path.isdir(sys.argv[1]):
            current_dir = sys.argv[1]
        else:
            print(f"Error: '{sys.argv[1]}' is not a valid directory")
            sys.exit(1)
    
    # Confirm before proceeding
    print(f"This will delete all image files except the first one in each subfolder of: {current_dir}")
    print("Proceed? (y/n)")
    
    confirmation = input().strip().lower()
    if confirmation != 'y' and confirmation != 'yes':
        print("Operation canceled.")
        sys.exit(0)
    
    # Process the directory
    keep_first_image_only(current_dir)