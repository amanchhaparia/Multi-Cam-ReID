import cv2
import torch
import torchreid
from torchreid.utils import FeatureExtractor

import numpy as np
if __name__ == "__main__":
    vs = cv2.VideoCapture("/home/ppspr/Downloads/view-IP1.mp4")
    vs.set(cv2.CAP_PROP_FPS, 25)
    ano = open('ano4.txt','r')
    out = open('output.txt', 'w')
    cnt = 0
    pfrm = 0
    frm = 0
    while True:
        # Read frame 
        ret, frame = vs.read()
        if ret!=True:
            break    
        pfrm = frm
        line = ano.readline().split()
        while (frm==pfrm and cnt != 23535):          
            cnt+=1
            pfrm = frm
            l = ano.readline()
            line = l.split()
            frm = int(line[5])
            bbox = [int(line[1]), int(line[2]), int(line[3]), int(line[4])]
            id = int(line[0])
            left, top, right, bottom = bbox
            lost = int(line[6])
            generated = int(line[8])
            occ = int(line[7])
            if lost == 0 and occ == 0:
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), [255,0,0], 1)
                cv2.putText(frame, f"id : {id} ", (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
                cropped_img = frame[top:bottom, left:right]
                extractor = FeatureExtractor(
                    model_name='densenet201',
                    device='cpu'
                )
                features = extractor(cropped_img)
                bb = l[:-1]+" "+features
                lineWithFeature = ' '.join(map(str, features))
                out.write(lineWithFeature)
        cv2.imshow("result",frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            vs.release()
            break
    ano.close()
    out.close()