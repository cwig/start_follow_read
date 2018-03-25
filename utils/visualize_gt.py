import cv2
import numpy as np
import sys
import json
import math

def main():
    image_path = sys.argv[1]
    json_path = sys.argv[2]
    output_image_path = sys.argv[3]

    img = cv2.imread(image_path)
    with open(json_path) as f:
        data = json.load(f)

    for i, d in enumerate(data):
        print i
        print d['gt']
        print d['pred']
        print "---"
        prev_pt = None
        for pt in d['lf']:

            x = int((pt['x0'] + pt['x1'])/2.0)
            y = int((pt['y0'] + pt['y1'])/2.0)

            cv2.circle(img,(x,y), 4, (0,0,255), -1)
            if prev_pt is not None:
                cv2.line(img,(x,y), prev_pt, (0,0,255), 2)
            prev_pt = (x,y)

        x0 = d['sol']['x0']
        x1 = d['sol']['x1']
        y0 = d['sol']['y0']
        y1 = d['sol']['y1']

        dx = x0-x1
        dy = y0-y1

        d = math.sqrt(dx**2 + dy**2)/2

        mx = (x0+x1)/2.0
        my = (y0+y1)/2.0

        theta = -math.atan2(dx, -dy)

        cv2.circle(img,(int(mx),int(my)), int(d), (255,0,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,str(i),(int(mx),int(my)), font, 1,(255,255,255),2,cv2.LINE_AA)
        

    cv2.imwrite(output_image_path, img)

if __name__ == "__main__":
    main()
