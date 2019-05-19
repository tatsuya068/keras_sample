
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

img = image.load_img('Cat.jpg')
img = np.array(img)
# plt.imshow(img)
# plt.show()

datagen = image.ImageDataGenerator(rotation_range = 30)
x = img[np.newaxis]
datagen_1 = datagen.flow(x,batch_size=1)
plt.figure(figsize=(8,8))


for i in range(9):
    batch = next(datagen_1)
    img_1 = batch[0].astype(np.uint8)
    plt.subplot(3,3,i+1)
    plt.imshow(img_1)
plt.show()


