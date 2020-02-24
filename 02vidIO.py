import cv2

# 读视频流，0表示读摄像头，路径表示读已有的视频
vid = cv2.VideoCapture(0)

# 写视频流，参数1为存储路径，参数2为加密方式，参数3为帧数，参数4为分辨率元组（需要和实际图片一致）
vidSave = cv2.VideoWriter('.\\temp\\vidSave.avi', cv2.VideoWriter_fourcc(*'XVID'), 60.0, (640, 480))

while vid.isOpened():
    ret, frame = vid.read()
    if ret:
        # cv2.flip 图片翻转，0上下偏转，1左右偏转，默认读视频流时已经左右偏转过一次，这里需要转回来
        frame = cv2.flip(frame, 1)

        # 按帧写入
        vidSave.write(frame)

        cv2.imshow('vid', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

vid.release()
vidSave.release()
cv2.destroyAllWindows()
