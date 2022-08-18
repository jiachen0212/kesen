import os


def is_image(file_path):
    _, ext = os.path.splitext(file_path)
    if ext in ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'):
        return True
    return False


def for_each_image(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext in ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'):
                yield os.path.join(root, file)
