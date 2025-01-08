import requests
# fetches HTML content and downloads images
from bs4 import BeautifulSoup
# parses HTML files to find elements
import os
# folder/file operations 

'''
Script used to parse through Wikipedia Images and find species photos 
Wikipedia Images was used due to its easy accessibility (not requiring an API Key)
iNaturalist can also be accessed for manual 

Image Loading isn't 100% all the time, so some manual sorting needs to be done.
This process should be minimal as long as you use the scientific name.

Arguments Used:
    species (loaded from our own database of plant species and names)
    outputDir (using OS to give the folder path that they will be stored in.)
'''

def scrapeImages(species, outputDir):
    '''
    ---- Query and Load Wikipedia Commons Url ----
    '''
    search_query = species.replace(" ","+")
    # replaces all spaces in the species name with +. This is what is required when you need to do querry searches. 
    # this puts the code in the proper format for a web url
    base_url = "https://commons.wikimedia.org"
    # shortcut for wikipedia commons
    url = f"{base_url}/w/index.php?search={search_query}&title=Special%3ASearch&go=Go&type=image"
    # url construction to load images
    print(f"Scraping images from: {url}")
    # test to see if we are pulling the right url

    '''
    ---- Fetch data from the url ----
    '''
    response = requests.get(url)
    # sends get request to the url
    if response.status_code != 200:
    # checks to see if HTTP 200 OK fulfilled 
    # 200 is a status code showing that the 
        print(f"Failed to access {url}")
        return
        # terminates the function if request fails and displays in terminal

    # parse HTML content
    soup = BeautifulSoup(response.text, "html.parser")
