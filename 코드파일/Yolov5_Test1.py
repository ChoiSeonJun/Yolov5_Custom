import tkinter as tk
from tkinter import messagebox, scrolledtext
import torch
import cv2
from PIL import Image, ImageTk


class YOLOv5App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv5 Object Detection")
        self.root.geometry("1000x600")  # 첫 기본창 크기 설정

        # 왼쪽 상단: 버튼 영역
        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        self.start_button = tk.Button(self.button_frame, text="시작", font=("Arial", 12), command=self.start_detection)
        self.start_button.pack(side="left", padx=5)

        self.exit_button = tk.Button(self.button_frame, text="종료", font=("Arial", 12), command=self.close)
        self.exit_button.pack(side="left", padx=5)

        # 웹캠 출력 영역 (초기 크기 조정)
        self.video_frame = tk.Frame(self.root, width=650, height=500, bg="white", relief="solid", bd=2)  # 초기 크기
        self.video_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

        self.video_label = tk.Label(self.video_frame, text="화면 준비 중...", font=("Arial", 14), bg="white")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")  # 중앙에 텍스트 배치

        # 오른쪽 하단: 로그 출력 영역
        self.right_frame = tk.Frame(self.root, width=300, height=500)
        self.right_frame.grid(row=1, column=1, padx=10, pady=10, sticky="se")

        self.log_label = tk.Label(self.right_frame, text="로그 출력", font=("Arial", 12))
        self.log_label.pack(pady=5)

        self.log_box = scrolledtext.ScrolledText(self.right_frame, width=40, height=10, state="disabled")
        self.log_box.pack()

        # YOLOv5 모델 초기화
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 기존 yolov5

        # 웹캠 및 실행 상태 초기화
        self.webcam = None
        self.running = False

    ## 웹캠이 잘 연결되어있는지 확인
    def start_detection(self):
        if self.webcam is None:
            self.webcam = cv2.VideoCapture(0)  # 웹캠 열기

            if not self.webcam.isOpened():
                messagebox.showerror("Error", "웹캠이 연결되어 있지 않습니다.")
                self.webcam = None
                return

            # 웹캠 해상도 설정
            # self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.running = True
        self.log_message("객체 감지를 시작합니다.")
        self.detect_objects()  ## YOLO시작

    ##YOLO 객체탐지 시작
    def detect_objects(self):
        if self.running and self.webcam.isOpened():
            # 웹캠에서 프레임 읽기
            ret, frame = self.webcam.read()
            if ret:
                # YOLOv5 모델로 객체 감지
                results = self.model(frame)
                output_frame = results.render()[0]

                # 감지 결과 로그 출력
                self.log_message("객체 감지 결과 업데이트 중...")
                for detection in results.pandas().xyxy[0].itertuples():
                    self.log_message(f"객체: {detection.name}, 신뢰도: {detection.confidence:.2f}")

                # OpenCV의 BGR 이미지를 RGB로 변환
                rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

                # PIL로 변환 후 tkinter에 표시
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            # 30ms 후 detect_objects 다시 호출 (프레임 업데이트)
            self.root.after(30, self.detect_objects)

    def close(self):
        # 감지 종료
        self.running = False
        if self.webcam is not None:
            self.webcam.release()
            self.webcam = None
        self.root.destroy()  # tkinter 창 닫기

    def log_message(self, message):
        # 로그 출력
        self.log_box.configure(state="normal")
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.configure(state="disabled")
        self.log_box.see(tk.END)  # 자동 스크롤


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv5App(root)
    root.mainloop()