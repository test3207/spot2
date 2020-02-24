import cv2

# imread('dir', {n}) n=0灰度图，n=1或者不填原图，n=-1α通道图
img = cv2.imread('.\\img\\img.jpg')
cv2.imshow('img', img)

cv2.waitKey()
cv2.destroyAllWindows()

# imwrite('dir', img)
