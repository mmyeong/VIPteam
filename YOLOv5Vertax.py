import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math


###라벨링 함수###
def setLabel(img, pts, label):
    (x, y, w, h) = cv2.boundingRect(pts)  # 바운딩 박스 좌표 추출
    pt1 = (x, y + 630)
    pt2 = (x + w, y + h)
    cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255))  # 텍스트 기입


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(
                            cls)], line_thickness=3)  # plot the bounding boxes

                        if names[int(cls)] == 'ThreeWayBlock' and conf >= 0.6:
                            ########################################################################################################################################
                            x1, y1, x2, y2 = int(xyxy[0].item()), int(xyxy[1].item()), int(xyxy[2].item()), int(
                                xyxy[3].item())  # bounding box

                            middle_x, middle_y = (x1 + x2) / 2, (y1 + y2) / 2  # creating middle point in bounding box

                            x11, x22, y11, y22 = int(middle_x) - 30, int(middle_x) + 30, int(middle_y) - 30, int(
                                middle_y) + 30  # small rectangular box from middle point

                            img = im0
                            if img is None:
                                print('Image load failed!')
                                return

                            region = im0[y11:y22,
                                     x11:x22]  # crop the region created from small rectangular from middle point
                            b, g, r = np.mean(region, axis=(0, 1)).round(
                                2)  # generating the average values of bgr in selected region
                            # b, g, r = np.median(region, axis=(0, 1)).round(2)
                            # print([b,g,r])

                            kernel = np.ones((33, 33), np.uint8)  # 커널값
                            # creating range from average bgr

                            lower = (b - 10, g - 10, r - 10)  # BGR minimum 범위
                            higher = (b + 10, g + 10, r + 10)  # BGR Maximum 범위
                            cropped_img = im0[y1:y2, x1:x2]  # 관심영역 추출
                            dst = cv2.inRange(cropped_img, lower, higher)  # 관심영역안의 원하는 색상범위로 이진화처리
                            closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)  # 모폴로지 클로즈연산
                            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)  # 모폴로지 오프닝 연산

                            ################################contours####################################################
                            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL,
                                                           cv2.CHAIN_APPROX_NONE)  # findContours 함수로 외곽선 추출
                            for pts in contours:  # 노이즈 예외처리
                                if cv2.contourArea(pts) < 2000:
                                    continue
                                approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.01, True)  # 외곽선 근사화 처리
                                # print("approx : ", approx) #근사화 값 print로 확인용

                                vtc = len(approx)  # 꼭짓점 개수 추출
                                for i in range(approx.shape[0]):
                                    pt = (int(approx[i, 0, 0]),  # 꼭짓점 개수만큼 for문을 돌면서 좌표값을 pt에 저장
                                          int(approx[i, 0, 1]))
                                    cropped_img = im0[y1:y2, x1:x2]  # 욜로 바운딩박스에서 추출한 관심영역 좌표값 추출
                                    cv2.circle(cropped_img, pt, 5, (0, 0, 255), 2)  # 추출한 꼭짓점을 시각화하기위해 Circle함수 사용
                                    # cv2.circle(opened, pt, 5, (0, 0, 255), 2)
                                if vtc == 4:
                                    # setLabel(img, pts, 'GoStraight')
                                    setLabel(img, pts, '')
                                elif vtc == 12:  # 사거리의 경우 꼿짓점 12
                                    print("사거리입니다.")
                                    # speak('사거리입니다.')#TTS 사용시 출력할 Speak
                                    # setLabel(img, pts, 'ThreeWayBlock')
                                    setLabel(img, pts, '')
                                elif vtc == 8:  # 사거리의 경우 꼭짓점 8
                                    print("삼거리입니다.")
                                    # speak('삼거리입니다.')#TTS 사용시 출력할 Speak
                                    # setLabel(img, pts, 'ThreeWayBlock')
                                    setLabel(img, pts, '')
                            print(vtc)
                            Canny_HSV = cv2.Canny(opened, 50, 150)  # canny edge

                            # cv2.imshow('Canny_', Canny_HSV)
                            # cv2.imshow('Gaussian',opened)
                            cv2.imshow('img', img)
                            cv2.waitKey()
                            cv2.destroyAllWindows()
                    # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)# plot the bounding boxes

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                # cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:  # image 저장 함수
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')  # 한 프레임당 처리 시간 print


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='last.pt',
                        help='model.pt path(s)')  # last.pt : 학습된 YOLOv5 모델명
    parser.add_argument('--source', type=str, default='example0526.jpg',
                        help='source')  # default에서 '0' 으로 할시 Webcam으로 실시간 영상처리 가능
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
