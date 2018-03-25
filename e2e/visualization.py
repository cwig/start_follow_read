import numpy as np
import cv2

def draw_output(out, img):
    img = img.copy()

    for j in xrange(out['lf'][0].shape[0]):
        begin = out['beginning'][j]
        end = out['ending'][j]

        last_xy = None
        # for i in xrange(len(out['lf'])):
        begin_f = int(np.floor(begin))
        end_f = int(np.ceil(end))
        for i in xrange(begin_f, end_f+1):

            if i == begin_f:
                p0 = out['lf'][i][j].mean(axis=1)
                p1 = out['lf'][i+1][j].mean(axis=1)
                t = begin - np.floor(begin)
                p = p0 * (1 - t) + p1 * t

            elif i == end_f:

                p0 = out['lf'][i-1][j].mean(axis=1)
                if i != len(out['lf']):
                    p1 = out['lf'][i][j].mean(axis=1)
                    t = end - np.floor(end)
                    p = p0 * (1 - t) + p1 * t
                else:
                    p = p0
            else:
                p =  out['lf'][i][j].mean(axis=1)


            x = p[0]
            y = p[1]

            x = int(x)
            y = int(y)

            color = (0,0,0)
            cv2.circle(img,(x,y), 4, color, -1)

            if last_xy is not None:
                cv2.line(img, (x,y), last_xy, color, 2)

            last_xy = (x,y)

    for i in xrange(out['sol'].shape[0]):

        p = out['sol'][i]

        c = int(255 * p[-1])
        color = (c,0,255-c)

        x = p[0]
        y = p[1]
        r = p[2]
        x_comp = np.cos(r)
        y_comp = -np.sin(r)
        s = p[3]

        rx = x + s * x_comp * 2
        ry = y + s * y_comp * 2

        rx2 = x - s * x_comp
        ry2 = y - s * y_comp

        rx = int(rx)
        ry = int(ry)

        rx2 = int(rx2)
        ry2 = int(ry2)

        x = int(x)
        y = int(y)
        scale = abs(int(s))

        # color = (0,0,255)

        cv2.circle(img,(x,y), int(scale), color, 2)
        cv2.circle(img,(x,y), 4, color, -1)
        cv2.arrowedLine(img, (x,y), (rx,ry), color, 2, tipLength=0.25)
        # cv2.line(img, (rx2,ry2), (rx,ry), color, 2)
        cv2.putText(img,str(i),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
    return img
