import requests
from io import BytesIO
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PIL import Image
import os

# Load environment variables from the specified path
load_dotenv("../../../")
image_api = os.getenv("IMAGE_API")

def image_url(movie_list):
    """
    Fetches movie poster images from the OMDb API using a list of movie IDs.

    Parameters:
    -----------
    movie_list : list of str
        A list of IMDb movie IDs for which the corresponding posters need to be fetched.

    Returns:
    --------
    list of PIL.Image.Image
        A list of PIL Image objects containing the retrieved movie posters.

    Notes:
    ------
    - Uses the OMDb API (`http://img.omdbapi.com/`) to retrieve images.
    - Requires a valid API key stored in an environment variable (`IMAGE_API`).
    - If an image cannot be retrieved (e.g., due to an invalid ID or API issues), a failure message is printed.
    - The function does not return any failed requests in the list.

    Example Usage:
    --------------
    ```python
    movie_ids = ["tt0111161", "tt0068646", "tt0071562"]  # Example IMDb movie IDs
    images = image_url(movie_ids)
    for img in images:
        img.show()  # Opens each image
    ```
    """
    images = []
    for id in movie_list:
        url = f'http://img.omdbapi.com/?i={id}&apikey={image_api}'
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            images.append(img)
        else:
            print(f"Failed to retrieve image from {url}")
    return images

def show_image(image_list):
    """
    Displays a list of images side by side using Matplotlib.

    Parameters:
    -----------
    image_list : list of PIL.Image.Image
        A list of images (PIL Image objects) to be displayed.

    Returns:
    --------
    None
        The function does not return any value but renders the images in a Matplotlib figure.

    Notes:
    ------
    - If only one image is provided, it is still displayed properly.
    - Axes are hidden for a clean display.
    - The figure size is set dynamically based on the number of images to maintain readability.

    Example Usage:
    --------------
    ```python
    images = image_url(["tt0111161", "tt0068646"])
    show_image(images)
    ```
    """
    # Plot images side by side
    fig, axes = plt.subplots(1, len(image_list), figsize=(15, 5))

    # If only one image, axes is not iterable
    if len(image_list) == 1:
        axes = [axes]

    for ax, img in zip(axes, image_list):
        ax.imshow(img)
        ax.axis("off")  # Hide axes

    plt.show()
