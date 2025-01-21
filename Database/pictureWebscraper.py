import requests
from bs4 import BeautifulSoup
import os

'''
Script used to parse through Wikipedia Images and find species photos 
Wikipedia Images was used due to its easy accessibility (not requiring an API Key)
iNaturalist can also be accessed manually

Image loading isn't 100% all the time, so some manual sorting needs to be done.
This process should be minimal as long as you use the scientific name.

Arguments Used:
    species (loaded from our own database of plant species and names)
    outputDir (using OS to give the folder path that they will be stored in.)
'''

def scrapeImages(species, outputDir):
    '''
    ---- Query and Load Wikipedia Commons Url ----
    '''
    search_query = species.replace(" ", "+")
    # replaces all spaces in the species name with +. This is what is required when you need to do query searches.
    base_url = "https://commons.wikimedia.org"
    url = f"{base_url}/w/index.php?search={search_query}&title=Special%3ASearch&go=Go&type=image"
    print(f"Scraping images from: {url}")

    '''
    ---- Fetch data from the url ----
    '''
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Failed to access {url}: {e}")
        return

    # parse HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    '''
    ---- Extract image URLs ----
    '''
    image_elements = soup.select(".searchResultImage img")  # Selects image elements on the page

    if not image_elements:
        print("No images found for the given species.")
        return

    image_urls = []
    for img in image_elements:
        img_url = img.get("src")
        if img_url and img_url.startswith("//"):
            img_url = "https:" + img_url
        if img_url:
            image_urls.append(img_url)
            print(f"Found image URL: {img_url}")

    '''
    ---- Save images to outputDir ----
    '''
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    for i, img_url in enumerate(image_urls):
        try:
            img_data = requests.get(img_url).content
            file_path = os.path.join(outputDir, f"{species.replace(' ', '_')}_{i+1}.jpg")
            with open(file_path, 'wb') as img_file:
                img_file.write(img_data)
            print(f"Saved image to: {file_path}")
        except Exception as e:
            print(f"Failed to download image from {img_url}: {e}")

# Example usage:
species_name = "Hibiscus brackenridgei"  # Replace with your test species
output_directory = "./species_images"  # Replace with your desired output directory
scrapeImages(species_name, output_directory)
