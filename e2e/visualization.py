def draw_output_highlight(out, img):
    img = img.copy()
#highlights = []
    #colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    colors = [(0.7,1,1),(1,0.7,1),(1,1,0.7),(0.85,0.85,1),(0.85,1,0.85),(1,0.85,0.85)]

    for j in xrange(out['lf'][0].shape[0]):
    highlight = np.ones(img.shape,dtype=np.float32)
        colorIdx=(j+2)%len(colors)
        begin = out['beginning'][j]
        end = out['ending'][j]

    pts_top=[]
    pts_bot=[]
        last_xy = None
        # for i in xrange(len(out['lf'])):
        begin_f = int(np.floor(begin))
        end_f = int(np.ceil(end))
        for i in xrange(begin_f, end_f+1):

            if i == begin_f:
                p0 = out['lf'][i][j]
                p1 = out['lf'][i+1][j]
                t = begin - np.floor(begin)
                p = p0 * (1 - t) + p1 * t
                pts_top.append([p[0][1],p[1][1]])
                pts_bot.append([p[0][0],p[1][0]])

            elif i == end_f:

                p0 = out['lf'][i-1][j]
                if i != len(out['lf']):
                    p1 = out['lf'][i][j]
                    t = end - np.floor(end)
                    p = p0 * (1 - t) + p1 * t
                else:
                    p = p0
                pts_top.append([p[0][1],p[1][1]])
                pts_bot.append([p[0][0],p[1][0]])
            else:
                #print out['lf'][i][j]
                pts_top.append([out['lf'][i][j][0][1],out['lf'][i][j][1][1]])
                pts_bot.append([out['lf'][i][j][0][0],out['lf'][i][j][1][0]])



	#colorIdx=(1+colorIdx)%len(colors)
            #cv2.circle(img,(x,y), 4, color, -1)

            #if last_xy is not None:
            #    cv2.line(img, (x,y), last_xy, color, 2)


        pts = np.array(pts_top+(pts_bot[::-1]),dtype=np.int32).reshape(-1,1,2)
        #print pts
        color = colors[colorIdx]
        cv2.fillPoly(highlight, [pts], color)
        #highlights.push(highlight)
        img = (img*highlight)

	#img = cv2.addWeighted(img, 1, src2, )
	img = img.astype(np.uint8)

    return img

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
