# AI Project: INBIS
### IN(내부) + BIS(버스정보시스템)

### Description
- 버스 내부 카메라를 이용하여 AI Face detection을 이용하여 버스 앞쪽의 인원수가 일정 수준을 초과할 시 뒤로 이동하도록 안내 메시지를 울린다.

### Member
- 소속: 국민대학교 소프트웨어학부, 경영정보학부
- 김명진, 이다은, 이도훈, 이정하, 조재오, 홍진욱

### Requirements
- Torch == 1.3.1
- Open CV == 4.1.1
- Numpy == 1.17.3
- Python == 3.7.3
- NVIDIA Driver == 440.44
- NVIDIA GPU == GeForce 940MX
- Linux CUDA == 10.2

## 실행방법
1. weights 가져오기
```
cd INBIS
wget https://pjreddie.com/media/files/yolov3.weights 
```

2. 실시간 웹캠 영상처리
```
python cam_demo.py
```

3. 영상 파일 처리
```
python video_demo.py [--video [VIDEO_FILE]]
```

4. 최종 결과물 실행
```
python main.py  [--option [OPTION]] [--front [FRONT]] [--back [BACK]]
    --option    Choose one option video/webcam/image(default : video)
    --front     Front file log to run detection upon except that option is webcam
    --back      Back file log to run detection upon except that option is webcam
```
