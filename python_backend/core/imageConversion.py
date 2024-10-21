import os 
import SimpleITK as sitk

# simple image converter script, that iterates through folder and converts non .jpeg files to .jpeg files
# converts and replaces the file inside of the folder, without creating new folder 
''' look into application with image web scraper '''
def image_converter(inputImageFolder):
    for filename in os.listdir(inputImageFolder):
        if filename.lower().endswith("png"):
            imgPath = os.path.join(inputImageFolder, filename)
            # makes sure that the image is loaded in proper formatting for processing
            img = sitk.ReadImage(imgPath)
            # required format conversion for jpgs
            # img = sitk.Cast(img, sitk.sitkUInt8)
            if img.GetPixelID() != sitk.sitkUInt8:
                # Cast the image to sitkUInt8 if necessary
                img = sitk.Cast(img, sitk.sitkUInt8)
            newFilePath = os.path.splitext(imgPath)[0] + '.jpg'  # or '.jpeg'
            # overwrites original file
            sitk.WriteImage(img, newFilePath)
            # remove original file if it's not already jpg
            if imgPath != newFilePath:
                os.remove(imgPath)

            print(f"Converted and replaced {filename} with JPEG")

inputImageFolder = os.path.join("python_backend", "core", "Pictures", "1test")
input2 = "/Users/mmorales25/Documents/GitHub/Pu-u-o-Manoa-App/python_backend/core/Pictures/1test"
print(inputImageFolder)
if not os.path.exists(inputImageFolder):
    print(f"Directory {inputImageFolder} does not exist.")
else:
    image_converter(inputImageFolder)

print(input2)
if not os.path.exists(input2):
    print(f"Directory {input2} does not exist.")
else:
    image_converter(input2)

print(os.getcwd())
# Example usage
# image_converter(inputImageFolder)
