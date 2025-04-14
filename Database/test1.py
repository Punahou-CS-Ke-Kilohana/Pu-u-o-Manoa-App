import json

# Open and read the JSON file
with open('/Users/dyee25/Documents/GitHub/Pu-u-o-Manoa-App/Database/plant_data.json', 'r') as file:
    data = json.load(file)

plants = []
# Iterate through the dictionary
for plant, details in data.items():
    # Add the common names to the list if they exist
    if 'Common Names' in details:
        # Extend the list with common names (assuming they are a list)
        plants.extend(details['Common Names'])

print(plants)
