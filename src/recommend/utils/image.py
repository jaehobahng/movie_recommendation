import requests
from io import BytesIO

def image_url(movie_list):
    images = []
    for id in movie_list:
        url = f'http://img.omdbapi.com/?i={id}&apikey=49b60587'
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            images.append(img)
        else:
            print(f"Failed to retrieve image from {url}")
    return images