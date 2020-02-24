# 绘制几何图形
# opencv采用BGR，和正常RGB顺序不同，传参转化时需要注意
import numpy as np
import cv2 as cv

# 宽度-1为绘制封闭图形及其内部；
# 原图左上角默认为坐标原点(0, 0)

# line 参数1为原图（可以直接用np三维数组代替），参数2、3为线段起止点，参数4为颜色，参数5为宽度
img = cv.line(np.zeros((512, 512, 3), np.uint8), (0, 0), (511, 511), (255, 0, 0), 5)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

# rectangle 原图，左上角、右下角、颜色、宽度
img = cv.rectangle(img, (100, 1), (412, 511), (0, 255, 0), 10)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

# circle 原图，圆心、半径、颜色、宽度
img = cv.circle(img, (23, 23), 20, (0, 0, 255), -1)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

# 其他图形暂时用不到