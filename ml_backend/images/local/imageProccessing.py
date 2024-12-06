import os

def get_image_paths(folder_path, output_file):
    with open(output_file, 'w') as f:
        # Walk through all folders and files
        index = 0  # Start the index from 0
        for root, dirs, files in os.walk(folder_path):
            has_image = False  # Flag to check if the folder contains images
            for file in files:
                # Check if file is an image
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    has_image = True
                    # Get the relative file path
                    relative_path = os.path.relpath(os.path.join(root, file), folder_path)
                    # Write the relative path and index to the output file
                    f.write(relative_path + ', ' + str(index) + '\n')
                    print(relative_path + '\n')
            # Increment index only if there was at least one image in the folder
            if has_image:
                index += 1

# Replace with your actual folder path and desired output file path
folder_path = 'python_backend/Pictures'  # Change to your folder's path
output_file = 'python_backend/Pictures/picturePaths.txt'  # Output text file

get_image_paths(folder_path, output_file)
