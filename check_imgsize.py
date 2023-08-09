from PIL import Image

im = Image.open('/storage/locnx/train2014/COCO_train2014_000000581880.jpg')

width, height = im.size

print(width)
print(height)