import torch
import os
import cv2
import time
import shutil
import argparse
import platform
from numpy import random
from pathlib import Path
import torch.backends.cudnn as cudnn

from models.models import *
from utils.datasets import *
from utils.general import *
from interface_calibrations import *

from utils.plots import plot_one_box
from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import (check_img_size, non_max_suppression, 
                           apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque

def draw_boxes(img, bbox, object_id, identities=None, offset=(0, 0)):

    height, _, _ = img.shape 
    calilbration_constant = .01

    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        box_height = (y2-y1)

        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        id = int(identities[i]) if identities is not None else 0

        if id not in data_deque:  
          data_deque[id] = deque(maxlen= opt.trailslen)

        color = compute_color_for_labels(object_id[i])
        label = '%s' % (names[object_id[i]])

        data_deque[id].appendleft(center)

        UI_box(box, img, label=label, color=color, line_thickness=2)

        for i in range(1, len(data_deque[id])):
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue

            thickness = int(np.sqrt(opt.trailslen / float(i + i)) * 1.5)
            
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

    return img

def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)


    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out) 
    os.makedirs(out) 
    half = device.type != 'cpu'  

    model = Darknet(cfg, imgsz).cuda()
    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])

    model.to(device).eval()
    
    if half:
        model.half()

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']) 
        modelc.to(device).eval()

    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True 
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    names = load_classes(names)
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  
    _ = model(img.half() if half else img) if device.type != 'cpu' else None 
    prevTime = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() 
        img /= 255.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):  
            if webcam:  
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  
            if det is not None and len(det):
                
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum() 
                    s += '%g %ss, ' % (n, names[int(c)]) 

                xywh_bboxs = []
                confs = []
                oids = []

                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    oids.append(int(cls))

                    if save_txt:  
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh)) 

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                
                outputs = deepsort.update(xywhs, confss, oids, im0)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]

                    draw_boxes(im0, bbox_xyxy, object_id,identities)

            print('%sDone. (%.3fs)' % (s, t2 - t1))

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.line(im0, (20,25), (127,25), [85,45,255], 30)
            cv2.putText(im0, f'FPS: {int(fps)}', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):
                    raise StopIteration

            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path: 
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release() 

                        fourcc = 'mp4v'  
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update: 
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  
    parser.add_argument('--output', type=str, default='inference/output', help='output folder') 
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    parser.add_argument('--trailslen', type=int, default=64, help='trails size (new parameter)')
        
    opt = parser.parse_args()
    print(opt)
    global names
    names = load_classes(opt.names)

    with torch.no_grad():
        if opt.update: 
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
