from PIL import Image
from defenitions import ROOT_DIR

home = ROOT_DIR


def create_thumbnail(file_name: str):
    image = Image.open(file_name)
    thumbnail_path = "{0}/thumbnails/{1}".format(home, file_name.split("/")[-1])

    size = 1280, image.size[1] * 1280 // image.size[0]
    image.thumbnail(size, Image.ANTIALIAS)
    image.save(thumbnail_path, "JPEG")

    return thumbnail_path
