from bs4 import BeautifulSoup
import requests
import os
import json

# 1.remove the thumb subdirectory 
# 2.remove everything after ... .svg/ 
# 3.this will give you the actual image and not just the thumbnail image

def get_soup(url, headers):
    response = requests.get(url, headers=headers)
    print(response)
    if response.status_code == 200:
        return BeautifulSoup(response.text, 'html.parser')
    else:
        print(f"Error fetching page: {response.status_code}")
        return None

query = "Ohia"  # You can change the query for the image here
image_type = "Action"
query = '+'.join(query.split())
url = f"https://www.google.com/search?q={query}&source=lnms&tbm=isch"

# Add the directory for your images here
DIR = "Pictures"
if not os.path.exists(DIR):
    os.mkdir(DIR)

header = {'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

soup = get_soup(url, header)
if soup is None:
    exit()

# This will hold the image URLs and types
ActualImages = []

# Google's image result structure has changed, so use a more general class for divs
for img_tag in soup.find_all("img"):
    img_url = img_tag.get("src")
    print(img_url)
    if img_url and "http" in img_url:
        ActualImages.append((img_url, "jpg"))  # Assuming JPG for simplicity

print(f"There are total {len(ActualImages)} images found")

# Creating a subdirectory for the images
query_dir = os.path.join(DIR, query.split()[0])
if not os.path.exists(query_dir):
    os.mkdir(query_dir)

# Save images
for i, (img_url, img_type) in enumerate(ActualImages):
    try:
        img_data = requests.get(img_url).content
        img_name = f"{image_type}_{i + 1}.{img_type}"
        img_path = os.path.join(query_dir, img_name)

        with open(img_path, 'wb') as f:
            f.write(img_data)
        print(f"Image saved: {img_path}")
    except Exception as e:
        print(f"Could not load image {img_url}")
        print(e)
