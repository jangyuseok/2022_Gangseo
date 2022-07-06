
import numpy as np
import cv2
import pandas as pd

cap = cv2.VideoCapture("1_1.mp4")
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = int(width)
height = int(height)
print(frames_count, fps, width, height)

history = 500
varThreshold = 40
detectShadow=False
sub = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadow)  #배경 추정을 위한 MOG2 알고리즘

ret, frame = cap.read()  #이미지로 불러오기
ratio = 0.8  #비율 설정
image = cv2.resize(frame, (0, 0), None, ratio, ratio)  #설정한 비율로 이미지 사이징
width2, height2, channels = image.shape



while True:
    ret, frame = cap.read() 
    if ret: 
        image = cv2.resize(frame, (0, 0), None, ratio, ratio) 
        cv2.imshow("image", image)
        roi = image[:,250:500]  #관심지역

        grayROI = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  #이미지 흑백으로 변환
        cv2.imshow("gray", grayROI)
        gmask = sub.apply(grayROI)
        cv2.imshow("gmask", gmask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  #형태변환 적용
        closing = cv2.morphologyEx(gmask, cv2.MORPH_CLOSE, kernel)  #Closing 기법
        cv2.imshow("closing", closing) 
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel) #Opening 기법 
        cv2.imshow("opening", opening) 
        dilation = cv2.dilate(opening, kernel)   #팽창 1회차
        cv2.imshow("dilation", dilation)   
        dilation2 = cv2.dilate(dilation, kernel)    #팽창 2회차
        cv2.imshow("dilation2", dilation2) 
        _, final = cv2.threshold(dilation2, 220, 255, cv2.THRESH_BINARY) #그림자 제거

        cv2.line(image, (300,150), (450,150), (0,0,255), 2) #주의선(빨강)
        # creates contours
        # cv2.imshow('bins',bins)
        contours, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        #윤곽선 인식 최소 범위
        minarea = 400
        #윤곽선 인식 최대 범위
        maxarea = 50000
                
        for i in range(len(contours)):  # 프레임의 모든 윤곽선 표시
                area = cv2.contourArea(contours[i])  # 윤곽선 범위
                if minarea < area < maxarea:  # 윤곽선 최대 최소 범위
                    # 윤곽선 중심점 계산
                    cnt = contours[i]

                    #cv2.moments 함수를 통해 이미지 모멘트를 계산하고 이를 딕셔너리 형태로 담아 리턴한다. 반환하는 모멘트는 총 24개로 10개의 위치 모멘트, 7개의 중심 모멘트, 7개의 정규화된 중심 모멘트로 이루어져 있다.
                    # - 공간 모멘트 (Spatial Moments) : m00, m10, m01, m20, m11, m02, m30, m21, m12, m03
                    # - 중심 모멘트 (Central Moments) : mu20, mu11, mu02, mu30, mu21, mu12, mu03
                    # - 평준화된 중심 모멘트 (Central Normalized Moments) : nu20, nu11, nu02, nu30, nu21, nu12, nu03
                    M = cv2.moments(cnt)
                    #대표적으로 Contours의 중심점을 구하는 공식
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    x, y, w, h = cv2.boundingRect(cnt)
                    # 사각형 윤곽선 생성
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    #윤곽선 중앙 아래부분 좌표
                    xMid = int((x+ (x+w))/2) 

                    #윤곽선 중앙 아래부분 노란점 표시
                    cv2.circle(roi, (xMid, y+h), 5,(0,255,255))

                    #윤곽선 중앙 아래부분 노란점 해당 범위를 지나면 warning 사인 표시
                    if (y+h > 150):
                        cv2.putText(roi, str("WARNING"),(cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("countours", image)
    key = cv2.waitKey(20)
    if key == 27:
       break

cap.release()
cv2.destroyAllWindows()