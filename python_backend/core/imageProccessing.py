import os

def get_image_paths(folder_path, output_file):
    with open(output_file, 'w') as f:
        # Walk through all folders and files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Check if file is an image
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    # Get the relative file path
                    relative_path = os.path.relpath(os.path.join(root, file), folder_path)
                    # Write the relative path to the output file
                    f.write(relative_path + '\n')
                    print(relative_path + '\n')

# Replace with your actual folder path and desired output file path
folder_path = 'python_backend/core/Pictures'  # Change to your folder's path
output_file = 'python_backend/core/Pictures/images.txt'  # Output text file

get_image_paths(folder_path, output_file)
