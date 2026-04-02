import cv2
import numpy as np

img = cv2.imread("variant-4.jpeg")
b, g, r = cv2.split(img)
zeros = np.zeros_like(b)
blue_only = cv2.merge([b, zeros, zeros])

cv2.imshow("Blue Channel Only", blue_only)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("blue_channel_result.jpg", blue_only)