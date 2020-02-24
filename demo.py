import cv2 as cv
import numpy as np
import math

PI = 3.14
# 红色通道阈值
MIN_RED = 180
# 蓝色通道阈值
MIN_BLUE = 180
# 其他通道阈值，实测约在100
MAX_OTHER = 150
# 白色通道阈值
MIN_WHITE = 150
# 黑色通道阈值
MAX_BLACK = 100
# 最小识别半径
MIN_RADIUS = 15
# 最小识别面积，过小无法提取信息
MIN_AREA = PI * MIN_RADIUS ** 2
# 最小圆度，过小纠偏后损失过多精度
MIN_CIRCULARITY = 0.7
# 最小凸度
MIN_CONVEXITY = 0.6
# 红圈红点阈值 < 1
RATIO_RED = 0.8
# 校验圈蓝点阈值 < 0.91
RATIO_BLUE = 0.7
# 校验圈白点阈值 < 0.09
RATIO_WHITE = 0.03
# 最大识别特征点数
MAX_FEATURES = 500
# 最小相似率
GOOD_MATCH_PERCENT = 0.15

# bgr类型识别，红r、蓝b、白w、黑d、其他o


def recoType(point):
    b, g, r = point
    # print(b, g, r)
    if r > MIN_RED and b < MAX_OTHER and g < MAX_OTHER:
        return 'r'
    elif b > MIN_BLUE and r < MAX_OTHER and g < MAX_OTHER:
        return 'b'
    elif b > MIN_WHITE and g > MIN_WHITE and r > MIN_WHITE:
        return 'w'
    elif b < MAX_BLACK and g < MAX_BLACK and r < MAX_BLACK:
        return 'd'
    else:
        return 'o'

# 返回圆点集


def circlePointSet(center, radius):
    ret = set([])
    cx, cy = int(center[0]), int(center[1])
    x0, x1 = math.floor(cx - radius), math.ceil(cx + radius)
    y0, y1 = math.floor(cy - radius), math.ceil(cy + radius)
    px, py = x0, y0
    while py < y1:
        px = x0
        while px < x1:
            if (px - cx) ** 2 + (py - cy) ** 2 < radius ** 2:
                ret.add((px, py))
            px += 1
        py += 1
    return ret

# 检查基本校验圈是否满足


def baseCheck(center, radius, frame):
    baseLength = radius / 3
    radiusRed = baseLength * 2
    setRed = circlePointSet(center, radiusRed)
    ret = True
    print('baseLength is:', baseLength)
    print('radiusRed is:', radiusRed)
    print('setRed length is', len(setRed))
    print(sum(map(lambda p: 1 if recoType(
        frame[p[0]][p[1]]) == 'r' else 0, setRed)) / len(setRed))
    print(sum(map(lambda p: 1 if recoType(
        frame[p[0]][p[1]]) == 'b' else 0, setRed)) / len(setRed))
    print(sum(map(lambda p: 1 if recoType(
        frame[p[0]][p[1]]) == 'w' else 0, setRed)) / len(setRed))
    print(sum(map(lambda p: 1 if recoType(
        frame[p[0]][p[1]]) == 'd' else 0, setRed)) / len(setRed))
    print(sum(map(lambda p: 1 if recoType(
        frame[p[0]][p[1]]) == 'o' else 0, setRed)) / len(setRed))
    if sum(map(lambda p: 1 if recoType(frame[p[0]][p[1]]) == 'r' else 0, setRed)) / len(setRed) < RATIO_RED:
        ret = False

    setSpot = circlePointSet(center, radius)
    setCheckCircle = setSpot - setRed

    print('setCheckCircle length is', len(setCheckCircle))
    print(sum(map(lambda p: 1 if recoType(
        frame[p[0]][p[1]]) == 'r' else 0, setCheckCircle)) / len(setCheckCircle))
    print(sum(map(lambda p: 1 if recoType(
        frame[p[0]][p[1]]) == 'b' else 0, setCheckCircle)) / len(setCheckCircle))
    print(sum(map(lambda p: 1 if recoType(
        frame[p[0]][p[1]]) == 'w' else 0, setCheckCircle)) / len(setCheckCircle))
    print(sum(map(lambda p: 1 if recoType(
        frame[p[0]][p[1]]) == 'd' else 0, setCheckCircle)) / len(setCheckCircle))
    print(sum(map(lambda p: 1 if recoType(
        frame[p[0]][p[1]]) == 'o' else 0, setCheckCircle)) / len(setCheckCircle))
    if sum(map(lambda p: 1 if recoType(frame[p[0]][p[1]]) == 'b' else 0, setCheckCircle)) / len(setCheckCircle) < RATIO_BLUE:
        ret = False
    if sum(map(lambda p: 1 if recoType(frame[p[0]][p[1]]) == 'w' else 0, setCheckCircle)) / len(setCheckCircle) < RATIO_WHITE:
        ret = False
    return ret

def alignImages(im1, im2):
 
    im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    
    orb = cv.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    
    imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv.imwrite('matches.jpg', imMatches)
    
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    h, mask = cv.findHomography(points1, points2, cv.RANSAC)
    
    height, width, channels = im2.shape
    im1Reg = cv.warpPerspective(im1, h, (width, height))
    
    return im1Reg, h

# 画出匹配转换图形
# 1、预处理输入图（划掉，预处理直接在上一层做）
# 2、orb识别特征点（sift算法专利今年6月才到期，在此之前可能有专利风险）
# 3、输出识别结果

def drawAlign():
    refFilename = 'example.jpg'
    imReference = cv.imread(refFilename, cv.IMREAD_COLOR)
    
    imFilename = 'part.png'
    im = cv.imread(imFilename, cv.IMREAD_COLOR)

    imReg, h = alignImages(im, imReference)
    
    outFilename = 'aligned.jpg'
    cv.imwrite(outFilename, imReg)
    
    print('Estimated homography : \n',  h)


# 帧处理
# 1、通过blobDetect找到圆心、半径
# 可能返回多个结果
# 2、对每一个结果手动校验baseCheck，蓝白环算点数调阈值
# 3、纠偏算法
# 4、尝试解码
# 返回值为灰度图、结果图；中间计算过程都直接打印在每一帧上；


def deal(frame):
    flag = False
    frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    params = cv.SimpleBlobDetector_Params()
    # params.filterByArea = True
    # params.minArea = MIN_AREA
    params.filterByCircularity = True
    params.minCircularity = MIN_CIRCULARITY
    params.filterByConvexity = True
    params.minConvexity = MIN_CONVEXITY
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(frame)
    redpoints = []
    # keypoint中有用的属性只有坐标pt和直径size
    # angle class_id octave pt response size
    # -1.0 -1 0 (361.3153076171875, 300.195556640625) 0.0 14.83292007446289
    for keypoint in keypoints:
        point = frame[int(keypoint.pt[1])][int(keypoint.pt[0])]
        if keypoint.size > MIN_RADIUS * 2 and recoType(point) == 'r' and baseCheck((keypoint.pt[1], keypoint.pt[0]), keypoint.size / 2, frame):
            # print(keypoint.pt[1], keypoint.pt[0])
            redpoints.append(keypoint)
            flag = True
            # 截取较小部分的图片
            cutRadius = int(keypoint.size * 10 / 3 / 2 * 1.1)
            cv.imwrite('part.png', frame[int(keypoint.pt[1]) - cutRadius : int(keypoint.pt[1]) + cutRadius, int(keypoint.pt[0]) - cutRadius : int(keypoint.pt[0]) + cutRadius])
    frameKeypoints = cv.drawKeypoints(frame, redpoints, np.array(
        []), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    frameKeypoints = cv.drawKeypoints(frameKeypoints, redpoints, np.array(
        []), (0, 0, 255))
    frameCanny = cv.Canny(frame, 600, 100, 3)

    if flag:
        drawAlign()

    return frameGray, frameKeypoints, frameCanny, cv.cvtColor(frame, cv.COLOR_BGR2HLS), flag


def __main__():
    vid = cv.VideoCapture(0)
    vid.set(3, 1280)
    vid.set(4, 720)
    while vid.isOpened():
        ret, frame = vid.read()
        frame = cv.flip(frame, 1)

        frameGray, frameDetect, frameCanny, frameHsv, flag = deal(frame)

        # cv.imshow('vidgrey', frameGray)
        cv.imshow('viddetect', frameDetect)
        # cv.imshow('vidcanny', frameCanny)
        # cv.imshow('vidhsv', frameHsv)

        if flag:
            cv.imwrite('red.png', frame)
            cv.waitKey(0)

        waitKey = cv.waitKey(1)

        if waitKey & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break
        elif waitKey & 0xFF == ord('c'):
            cv.imwrite('cut.png', frame)
            cv.waitKey(0)


__main__()
