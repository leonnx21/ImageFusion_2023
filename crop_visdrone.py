from PIL import Image
import os


def crop_func():

    new_width = 640
    new_height = 512

    path = '/storage/locnx/VisDrone/test/Test2/IR'
    image_name = sorted(os.listdir(path))
    save_path = '/storage/locnx/VisDrone/test/Test2/IR'

    for i, name in enumerate(image_name):
        image_path = os.path.join(path, name)

        im = Image.open(image_path)
        width, height = im.size   # Get dimensions

        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        # Crop the center of the image
        im = im.crop((left, top, right, bottom))
        # new_filename = str(i)+".png"

        image_save_path = os.path.join(save_path, name)
        
        print("saved:",image_save_path)

        im.save(image_save_path)

        im.close()

    print("Finished")

if __name__ == '__main__':
    crop_func()