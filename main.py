from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import cv2
from util import write_results, load_classes
from darknet import Darknet
from preprocess import prep_image
import numpy as np
import random 
import pickle as pkl
import argparse
import threading

def write(x, img, classes, colors):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    if label == "person":
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img, label

def check_person(max_val, loc):
    if loc == "front":
        threshold = 3
    else:
        threshold = 10

    if threshold < max_val:
        return "red"
    else:
        return "green"

def arg_parse():
    
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
    parser.add_argument("--option", dest = 'option', help = "video/webcam/image(default : video)", default = "video", type = str)
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()


def represent_case(orig_im, case):
    w, h, d = orig_im.shape
    filter_img = np.zeros((w, h, d), np.uint8)
    if case == "red":
        filter_img[:,:] = (0,0,255)
    else:
        filter_img[:,:] = (0,255,0)
    return cv2.addWeighted(orig_im, 0.8, filter_img, 0.2, 0)
    

def main():
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    #print("loc: ", loc)

    CUDA = torch.cuda.is_available()

    num_classes = 80

    CUDA = torch.cuda.is_available()
    
    bbox_attrs = 5 + num_classes
    
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
        
    model.eval()
    status = False
    option = args.option
    #if option == "webcam":
    #    if loc == "front":
    cap_front = cv2.VideoCapture(0)
    #    else:
    cap_back = cv2.VideoCapture(1)

    elif option == "video":
        videofile = args.video
        cap = cv2.VideoCapture(videofile)
    else:
        videofile = args.video
        cap = cv2.VideoCapture(videofile)
        status = True
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()  
    max_val_f = 0
    max_val_b = 0 
    tmp = 0
    while cap_back.isOpened() or cap_front.isOpened():
        ret_front, frame_front = cap_front.read()
        ret_back, frame_back = cap_back.read()
        if ret_front and ret_back:
            img_f, orig_im_f, dim_f = prep_image(frame_front, inp_dim)
            img_b, orig_im_b, dim_b = prep_image(frame_back, inp_dim)
            im_dim_f = torch.FloatTensor(dim_f).repeat(1,2)
            im_dim_b = torch.FloatTensor(dim_b).repeat(1,2)                        

            if CUDA:
                im_dim_f = im_dim_f.cuda()
                img_f = img_f.cuda()
                im_dim_b = im_dim_b.cuda()
                img_b = img_b.cuda()
            
            with torch.no_grad():   
                output_f = model(Variable(img_f), CUDA)
                output_b = model(Variable(img_b), CUDA)
            output_f = write_results(output_f, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            output_b = write_results(output_b, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            """    
            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            """
            im_dim_f = im_dim_f.repeat(output_f.size(0), 1)
            scaling_factor_f = torch.min(inp_dim_f/im_dim_f,1)[0].view(-1,1)
            im_dim_b = im_dim_b.repeat(output_b.size(0), 1)
            scaling_factor_f = torch.min(inp_dim_f/im_dim_f,1)[0].view(-1,1)
            
            output_f[:,[1,3]] -= (inp_dim_f - scaling_factor_f*im_dim_f[:,0].view(-1,1))/2
            output_f[:,[2,4]] -= (inp_dim_f - scaling_factor_f*im_dim_f[:,1].view(-1,1))/2
            
            output_F[:,1:5] /= scaling_factor_f
    
            for i in range(output_f.shape[0]):
                output_f[i, [1,3]] = torch.clamp(output_f[i, [1,3]], 0.0, im_dim[i,0])
                output_f[i, [2,4]] = torch.clamp(output_f[i, [2,4]], 0.0, im_dim[i,1])

            output_b[:,[1,3]] -= (inp_dim_b - scaling_factor_b*im_dim_b[:,0].view(-1,1))/2
            output_b[:,[2,4]] -= (inp_dim_b - scaling_factor_b*im_dim_b[:,1].view(-1,1))/2
            
            output_b[:,1:5] /= scaling_factor_b
    
            for i in range(output_b.shape[0]):
                output_b[i, [1,3]] = torch.clamp(output_b[i, [1,3]], 0.0, im_dim_b[i,0])
                output_f[i, [2,4]] = torch.clamp(output_f[i, [2,4]], 0.0, im_dim_f[i,1])
                        
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))
            #############33
            cnt_f = list(map(lambda x: write(x, orig_im_f, classes, colors)[1], output)).count("person")
            cnt_b = list(map(lambda x: write(x, orig_im_b, classes, colors)[1], output)).count("person")
            print("front person : " + str(cnt_f))
            print("back person : " + str(cnt_b))
            if max_va_f < cnt_f:
               max_val_f = cnt_f
            if max_val_b < cnt_b:
               max_val_b = cnt_b
            print("max_val_f : " + str(max_val_f))
            print("max_val_b : " + str(max_val_b))
            
            case_f = check_person(max_val_f, "front")
            case_b = check_person(max_val_b, "back")
            after_img_f = represent_case(orig_im_f, case_f)
            after_img_b = represent_case(orig_im_b, case_b)

            cv2.imshow(loc, after_img)
            if status:
                cv2.waitKey(-1)
            #cv2.imwrite('output/frame%04d' + loc + '.jpg' %(tmp), orig_im)
            tmp += 1
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
  
        else:
            break

if __name__ == '__main__':
    """
    t = threading.Thread(target=main, args=("back",))
    t.start()
    t = threading.Thread(target=main, args=("front",))
    t.start()
    """
    #loc = "front"
    main()
    
    

    
    

