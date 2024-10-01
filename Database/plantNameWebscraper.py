import requests
from bs4 import BeautifulSoup
import json
import re
import os

plantLinks = []

# gets links for all plants
for i in range(24):
    plantPage = f"index/page/{i + 1}/"
    r = requests.get(f'http://nativeplants.hawaii.edu/plant/{plantPage}')
    soup = BeautifulSoup(r.content, 'html.parser')
    linksHtml = soup.find_all(href=re.compile("plant/view"))
    for tag in linksHtml:
        plantLinks.append(tag["href"])

all_results = {}

for plant in plantLinks:
    r = requests.get(f'http://nativeplants.hawaii.edu{plant}')

    if r.status_code == 200:
        soup = BeautifulSoup(r.content, 'html.parser')

        headers = soup.find_all("p", class_="subheading")
        result = {}

        for header in headers:
            subheading_text = header.get_text(strip=True)
            # Get content for each header
            plantcontent = header.find_next('p', class_='plantcontent')

            if plantcontent:
                plantcontent_text = plantcontent.get_text(strip=True)
                # Parses out list items 
                list_items = plantcontent.find_all('li')
                if list_items:
                    plantcontent_list = [item.get_text(strip=True) for item in list_items]
                    result[subheading_text] = plantcontent_list
                else:
                    result[subheading_text] = plantcontent_text
    else:
        print(f"Could not get data for {plant}")

    all_results[plant.split("/")[-1]] = result

# Ensure the directory exists
output_dir = 'Database'
os.makedirs(output_dir, exist_ok=True)

# Save all data into a JSON file
output_file = os.path.join(output_dir, 'plant_data.json')
with open(output_file, 'w') as f:
    json_output = json.dumps(all_results, indent=4, ensure_ascii=False)
    f.write(json_output)
