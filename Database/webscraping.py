import requests
from bs4 import BeautifulSoup
import json

# name of all plants you want to scrape data for
plants = ["Abutilon_eremitopetalum", "Abutilon_incanum", "Abutilon_menziesii", "Dodonaea_viscosa", "Metrosideros_polymorpha"]

all_results = {}

for plant in plants:
    r = requests.get(f'http://nativeplants.hawaii.edu/plant/view/{plant}/')

    if r.status_code == 200:
        soup = BeautifulSoup(r.content, 'html.parser')

        headers = soup.find_all("p", class_="subheading")

        result = {}

        for header in headers:
            subheading_text = header.get_text(strip=True)
            # gets content for each header
            plantcontent = header.find_next('p', class_='plantcontent')

            if plantcontent:
                plantcontent_text = plantcontent.get_text(strip=True)
                # parses out list items 
                list_items = plantcontent.find_all('li')
                if list_items:
                    plantcontent_list = [item.get_text(strip=True) for item in list_items]

                    result[subheading_text] = plantcontent_list
                else:
                    result[subheading_text] = plantcontent_text
    else:
        print(f"Could not get data for the {plant}")

    all_results[plant] = result
# puts all data into json file
with open('Database/plant_data.json', 'w') as f:
    json_output = json.dumps(all_results, indent=4, ensure_ascii=False)
    f.write(json_output)
