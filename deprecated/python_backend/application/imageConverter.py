import os 
import SimpleITK as sitk

# simple image converter script, that iterates through folder and converts non .jpeg files to .jpeg files
# converts and replaces the file inside of the folder, without creating new folder 
''' look into application with image web scraper '''

def image_converter(inputImageFolder):
    for filename in os.listdir(inputImageFolder):
        if not filename.lower().endswith("jpeg"):
            imgPath = os.path.join(inputImageFolder, filename)
            # makes sure that the image is loaded in proper formatting for processing
            img = sitk.ReadImage(imgPath)
            
            # only cast if the image is scalar and not already sitkUInt8
                # this prevents errors and possible breaks in the code
                # files don't want to be cast if they are already in the proper format
            # grayscale images have only 1 component (with 0,255 unique values)
                # this makes them a scalar and 
                # however, we are checking if it is not a vector and has at least 3 components (RGB)    
            # pixel type being checked (not an RGB already)
                # reduces redundancy 
            # if not img.GetNumberOfComponentsPerPixel() > 1:
            #     if img.GetPixelID() != sitk.sitkUInt8:
            #         img = sitk.Cast(img, sitk.sitkUInt8)
            if img.GetNumberOfComponentsPerPixel() == 1:
                # Cast scalar images to sitkUInt8 if needed
                if img.GetPixelID() != sitk.sitkUInt8:
                    img = sitk.Cast(img, sitk.sitkUInt8)
            elif img.GetNumberOfComponentsPerPixel() == 3:
                # Cast multi-channel RGB images to sitkVectorUInt8 if not already
                if img.GetPixelID() != sitk.sitkVectorUInt8:
                    img = sitk.Cast(img, sitk.sitkVectorUInt8)


            newFilePath = os.path.splitext(imgPath)[0] + '.jpeg'  # or '.jpeg'
            # overwrites original file
            sitk.WriteImage(img, newFilePath)
            # remove original file if it's not already jpg
            if imgPath != newFilePath:
                os.remove(imgPath)

            print(f"Converted and replaced {filename} with jpeg")

# iterates through all folders in Pictures and converts non jpeg files to jpeg files
for foldername in os.listdir("Pictures"):
        inputImageFolder = os.path.join("Pictures", foldername)
        print(foldername)
        image_converter(inputImageFolder)


