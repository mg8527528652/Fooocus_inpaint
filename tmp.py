import os
import cv2

image_path = '/root/Fooocus_inpaint/data/input/1_Mountain.jpg'
mask_path = '/root/Fooocus_inpaint/data/mask/1.jpg'
imgs_save_path = '/root/Fooocus_inpaint/data/tmp/imgs'
masks_save_path = '/root/Fooocus_inpaint/data/tmp/masks'
os.makedirs(imgs_save_path, exist_ok=True)
os.makedirs(masks_save_path, exist_ok=True)

img = cv2.imread(image_path)
mask = cv2.imread(mask_path)

res = [512, 1024, 2048, 3200, 4096, 5000, 6000]

for i in res:
    img_resized = cv2.resize(img, (i, i))
    mask_resized = cv2.resize(mask, (i, i))
    cv2.imwrite(os.path.join(imgs_save_path, f'{i}.jpg'), img_resized)
    cv2.imwrite(os.path.join(masks_save_path, f'{i}.jpg'), mask_resized)