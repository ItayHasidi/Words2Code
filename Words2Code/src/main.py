# import easyocr
# import cv2
# from matplotlib import pyplot as plt
# import numpy as np
#
# imgPath1 = "images//text_example_1.jpg"
# imgPath2 = "images//text_example_2.jpg"
#
# reader = easyocr.Reader(['en'], gpu=False)
# result = reader.readtext(imgPath2)
# # print(result)


from PIL import Image
import pytesseract as pytes
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pytes.pytesseract.tesseract_cmd = 'C://Program Files//Tesseract-OCR//tesseract.exe'
# filename = 'images/text_example_2.jpg'
# filename = 'images//Capture.PNG'
filename = 'images//img_1.png'

img1 = np.array(Image.open(filename))

norm_img = np.zeros((img1.shape[0], img1.shape[1]))
img = cv2.normalize(img1, norm_img, 0, 255, cv2.NORM_MINMAX)
img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
img = cv2.GaussianBlur(img, (1, 1), 0)

text = pytes.image_to_string(img1)

print(text)


plt.imshow(img)
plt.show()