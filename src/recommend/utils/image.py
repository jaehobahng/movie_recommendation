import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

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

def show_image(image_list):
    # Plot images side by side
    fig, axes = plt.subplots(1, len(image_list), figsize=(15, 5))

    # If only one image, axes is not iterable
    if len(image_list) == 1:
        axes = [axes]

    for ax, img in zip(axes, image_list):
        ax.imshow(img)
        ax.axis("off")  # Hide axes

    plt.show()