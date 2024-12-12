import requests

def fetchFirstImageLink(species):
    base_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": species,
        "gsrlimit": 1,
        "prop": "imageinfo",
        "iiprop": "url"
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error accessing Wikimedia API: {e}")
        return

    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        image_info = page_data.get("imageinfo", [])
        if image_info:
            image_url = image_info[0].get("url")
            print(f"First image link: {image_url}")
            return
    print("No images found for the query.")

fetchFirstImageLink("Hawaiian honeycreeper")
