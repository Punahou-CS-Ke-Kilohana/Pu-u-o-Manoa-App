import json

with open('Database/plant_data.json') as f:
    data = json.load(f)

def find_by_hawaiian_name(data, hawaiian_name):
    for plant, info in data.items():
        if hawaiian_name.title() in info.get("Hawaiian Names", []):
            return {
                "Hawaiian Names": info.get("Hawaiian Names", "N/A"),
                "Common Names": info.get("Common Names", "N/A"),
                "Endangered Species Status": info.get("Endangered Species Status", "N/A"),
                "Natural Range": info.get("Natural Range", "N/A"),
                "Genus": info.get("Genus", "N/A"),
                "Species": info.get("Species", "N/A"),
                "Distribution Status": info.get("Distribution Status", "N/A"),
            }
    return None

result = find_by_hawaiian_name(data, "kou")

if result:
    print(json.dumps(result, indent=4, ensure_ascii=False))
else:
    print("Hawaiian name not found.")
