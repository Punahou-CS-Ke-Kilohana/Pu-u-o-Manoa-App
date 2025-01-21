import requests
from bs4 import BeautifulSoup
import os

'''
Script used to parse through Wikipedia Images and find species photos 
Wikipedia Images was used due to its easy accessibility (not requiring an API Key)
iNaturalist can also be used to load images manually

All images are loaded with the quality from the wikipedia commons website
They should all be high quality, but because there is such a large amount of images being loaded, all of the quality isn't examined

Arguments Used:
    species (loaded from our own database of plant species and names)
    outputDir (using OS to give the folder path that they will be stored in.)
    imageNumber (amount of images you want to load)

base_url (where we are pulling images from) will remain as commons.wikimedia.org
do not change any of the initial wikipedia commons stuff 

Works by going to the specific wikipedia image page for each species.
This page contains a bunch of other pages with images and information.
In order to get the highest quality images, we can't access the thumbnail photo,
meaning we need to go into the individual image pages and take the high quality
images from there. After that, all the data is stored in a new folder.

Other notes: 
    cannot be run on school wifi
'''

def scrapeImageUrls(species, outputDir, imageNumber):
    '''
    wikipedia commons url query 
    '''
    search_query = species.replace(" ", "+")
    # species name is rearranged to model URL syntax
    base_url = "https://commons.wikimedia.org"
    # where all the images are accessed from
    # search_url = f"{base_url}/w/index.php?title=Special:Search&limit=500&offset=0&ns0=1&ns6=1&ns12=1&ns14=1&ns100=1&ns106=1&search={search_query}+filemime%3Aimage%2Fjpeg&advancedSearch-current={%22fields%22:{%22filetype%22:%22image/jpeg%22}}"
    search_url = (
        f"{base_url}/w/index.php?title=Special:Search&limit=500&offset=0&ns0=1&ns6=1&ns12=1&ns14=1&ns100=1&ns106=1&"
        f"search={search_query}+filemime%3Aimage%2Fjpeg&advancedSearch-current=%7B%22fields%22:%7B%22filetype%22:%22image/jpeg%22%7D%7D"
    )
    print(search_url)
    # "{base_url}/w/index.php?search={search_query}&title=Special%3ASearch&go=Go&type=image"
    # this is the url format used to access the specific wikipedia commons image page 
    print(f"Scraping image URLs from: {search_url}")



    '''
    fetch data from the search page
    '''
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to access {search_url}: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    # parse HTML content

    '''
    extract individual image page URLs
    '''
    links = soup.find_all("a", href=True)
    file_page_urls = [base_url + link["href"] for link in links if link["href"].startswith("/wiki/File:")]

    if not file_page_urls:
        print("No file page URLs found for the given species.")
        return

    '''
    extract image URLs
    '''
    main_image_urls = []
    # initialize list to store all of the URLs used in image loading
    for file_page_url in file_page_urls[:imageNumber]:
    # only takes the first ___ amount of images (depends on input for image_number)
        try:
            file_response = requests.get(file_page_url, headers=headers)
            # gets the individual page files
            file_response.raise_for_status()
            file_soup = BeautifulSoup(file_response.text, "html.parser")
            main_image = file_soup.find("div", class_="fullImageLink").find("a", href=True)
            if main_image:
                image_url = "https:" + main_image["href"] if main_image["href"].startswith("//") else main_image["href"]
                if image_url not in main_image_urls: 
                # used to avoid image duplicates
                # this only works to prevent duplicates in one bunch, and not duplicates in the folder
                    main_image_urls.append(image_url)
                    print(f"Image URL from {file_page_url}: {image_url}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to access {file_page_url}: {e}")

    return main_image_urls

def downloadImages(image_urls, species, outputDir):
    '''
    download images from URLs
    '''
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    # this is required to bypass the User-agreement policy (will fail to load if not here)

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    for i, img_url in enumerate(image_urls, 1):
        try:
            response = requests.get(img_url, stream=True, headers=headers)
            response.raise_for_status()
            image_name = f"{species.replace(' ', '_')}_{i}.jpg"
            image_path = os.path.join(outputDir, image_name)
            with open(image_path, 'wb') as img_file:
                for chunk in response.iter_content(1024):
                    img_file.write(chunk)
            print(f"Saved image: {image_path}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {img_url}: {e}")

species_name = [] # Replace with your test species
# Example usage:
for i in len(species_name)
    output_directory = f"./{species_name}"  # Replace with your desired output directory
    image_number = 1000
    image_urls = scrapeImageUrls(species_name, output_directory, image_number)
    if image_urls:
        downloadImages(image_urls, species_name, output_directory)
