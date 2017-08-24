import cv2
import numpy as np
from managers import WindowManager, CaptureManager


class project4(object):
    
    def __init__(self):
        # window
        self._Window_name = 'project4'
        self._trackbar_name = 'trackbar'
        self._trackbar_LHchannel = "LHchannel"
        self._trackbar_LSchannel = "LSchannel"
        self._trackbar_LVchannel = "LVchannel"
        self._trackbar_HHchannel = "HHchannel"
        self._trackbar_HSchannel = "HSchannel"
        self._trackbar_HVchannel = "HVchannel"

        self._canny_high = "high"
        self._canny_low = "low"

        # coordinate 
        self._centeral_coordinate = []
        self._object_count = 0

        # image 
        self._LHchannel = 0
        self._LSchannel = 0
        self._LVchannel = 112
        self._HHchannel = 179
        self._HSchannel = 100
        self._HVchannel = 255

        self._high = 200
        self._low = 100

        # data sign
        self._number_string = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self._MINAREA = 150.0

        # show 
        self._show_flag = False

        # import init
        self._windowManager = WindowManager(self._Window_name, self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, False)



    def Track_onChange(self, back):
        """change value"""
        self._LHchannel = cv2.getTrackbarPos(self._trackbar_LHchannel, self._trackbar_name)
        self._LSchannel = cv2.getTrackbarPos(self._trackbar_LSchannel, self._trackbar_name)
        self._LVchannel = cv2.getTrackbarPos(self._trackbar_LVchannel, self._trackbar_name)
        self._HHchannel = cv2.getTrackbarPos(self._trackbar_HHchannel, self._trackbar_name)
        self._HSchannel = cv2.getTrackbarPos(self._trackbar_HSchannel, self._trackbar_name)
        self._HVchannel = cv2.getTrackbarPos(self._trackbar_HVchannel, self._trackbar_name)

        #self._high = cv2.getTrackbarPos(self._canny_high, self._trackbar_name)
        #self._low = cv2.getTrackbarPos(self._canny_low, self._trackbar_name)


    def white_balance(self, img):
        Kb = 0
        Kg = 0
        Kr = 0

        (img_b, img_g, img_r) = cv2.split(img)

        img_b_avg = cv2.mean(img_b)[0]
        img_g_avg = cv2.mean(img_g)[0]
        img_r_avg = cv2.mean(img_r)[0]

        K = (img_b_avg + img_g_avg + img_r_avg)/3
        
        if img_b_avg != 0.0 or img_b_avg != 0.0 or img_b_avg != 0.0 :
            Kb = K/img_b_avg
            Kg = K/img_g_avg
            Kr = K/img_r_avg

        img_b = cv2.addWeighted(img_b,Kb,0,0,0)
        img_g = cv2.addWeighted(img_g,Kg,0,0,0)
        img_r = cv2.addWeighted(img_r,Kr,0,0,0)

        src = cv2.merge((img_b, img_g, img_r))

        return src


    def filter(self, src, low, high, ksize_open, ksize_close, flag):
        lowHSV = np.array(low)
        highHSV = np.array(high)
        src_Filter = cv2.inRange(src, lowHSV, highHSV)
        src_Filter = cv2.GaussianBlur(src_Filter, (5,5), 0)
        #src_Filter = cv2.medianBlur(src_Filter, 7)

        if flag == True:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(ksize_open, ksize_open))
            src_Filter = cv2.morphologyEx(src_Filter, cv2.MORPH_OPEN, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(ksize_close, ksize_close))
            src_Filter = cv2.morphologyEx(src_Filter, cv2.MORPH_CLOSE, kernel)

        return src_Filter




    def camcatch(self, frame):
        # load picture
        srcRGB = frame

        #src = self.white_balance(srcRGB)
        #gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        #gray = cv2.medianBlur(gray, 7)
        #canny = cv2.Canny(gray, self._low, self._high)


        # create ROI mask
        mask_ROI = np.zeros([480,640], np.uint8)

        # convert to hsv and equalize it
        hsv = cv2.cvtColor(srcRGB, cv2.COLOR_BGR2HSV)
        (img_h, img_s, img_v) = cv2.split(hsv)
        img_v = cv2.equalizeHist(img_v)
        hsv = cv2.merge((img_h, img_s, img_v))

        # catch image and set ROI
        lowHSV = [self._LHchannel, self._LSchannel, self._LVchannel]
        highHSV = [self._HHchannel, self._HSchannel, self._HVchannel]
        Thresh_src = self.filter(hsv, lowHSV, highHSV, 27, 13, True)
        Thresh_src, contours_back, hier = cv2.findContours(Thresh_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_back:
            #rect = cv2.minAreaRect(cnt)
            #box = np.int0(cv2.boxPoints(rect))
            #cv2.drawContours(Thresh_back, [box], 0, (255,255,255), 3)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(Thresh_src, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.rectangle(srcRGB, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.circle(srcRGB, (x, y+h), 5, (255, 255, 255), -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21, 21))
        Thresh_back = cv2.morphologyEx(Thresh_src, cv2.MORPH_CLOSE, kernel)                         # get back image

        mask = np.zeros([482,642], np.uint8)
        retval, Thresh_object, mask, rect = cv2.floodFill(Thresh_back.copy(), mask, (0,0), 255)     # get object image
        Thresh_object, contours_object, hier = cv2.findContours((255 - Thresh_object), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self._object_count = 0
        self._centeral_coordinate = []
        for cnt in contours_object:
            Length = cv2.arcLength(cnt, 0)
            if Length < self._MINAREA:
                continue
            if Length >= self._MINAREA:
                self._object_count += 1

                M = cv2.moments(cnt)
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])

                rect_object = cv2.minAreaRect(cnt)
                box_object = np.int0(cv2.boxPoints(rect_object))
                cv2.drawContours(srcRGB, [box_object], 0, (0,0,0), 3)

                self._centeral_coordinate.append((cX, cY))
                cv2.circle(srcRGB, (cX, cY), 5, (255, 255, 255), -1)

                cv2.putText(srcRGB, self._number_string[self._object_count - 1], (cX - 10, cY - 10),
		                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        if self._show_flag:
            print('\nThe (0,0) coordinate : ', (x, y+h))
            for cnt in range(self._object_count):
                (cX,cY) = self._centeral_coordinate[cnt]
                print('\nThe', (cnt+1), 'object :')
                print('\ncenteral_coordinate : ', (abs(cX-x),abs(cY-(y+h))))

            self._show_flag = False


        #cv2.imshow('src', src)
        #cv2.imshow('gray', gray)
        #cv2.imshow('canny', canny)
        #cv2.imshow('Thresh_back', Thresh_back)
        #cv2.imshow('Thresh_object', Thresh_object)
        #cv2.imshow('Thresh_src', Thresh_src)
     



    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            
            if frame is not None:

                self.camcatch(frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()


    def onKeypress(self, keycode):
        """Handle a keypress.
        
        space  -> Take a screenshot.
        tab    -> Start/stop recording a screencast.
        escape -> Quit.
        
        """
        if keycode == 32: # space
            self._captureManager.writeImage('screenshot.png')

        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo(
                    'screencast.avi')
            else:
                self._captureManager.stopWritingVideo()

        elif keycode == 27: # escape
            self._windowManager.destroyAllWindow()
            print('\n lowHSV :', self._LHchannel, self._LSchannel, self._LVchannel)
            print('\n highHSV :', self._HHchannel, self._HSchannel, self._HVchannel)

            
        elif keycode == 115: # s
            print('\npress s -- show : show data')
            self._show_flag = True

        elif keycode == 116: # t
            cv2.namedWindow(self._trackbar_name)    
            cv2.createTrackbar(self._trackbar_LHchannel, self._trackbar_name, self._LHchannel, 180, self.Track_onChange)
            cv2.createTrackbar(self._trackbar_HHchannel, self._trackbar_name, self._HHchannel, 180, self.Track_onChange)
            cv2.createTrackbar(self._trackbar_LSchannel, self._trackbar_name, self._LSchannel, 255, self.Track_onChange)
            cv2.createTrackbar(self._trackbar_HSchannel, self._trackbar_name, self._HSchannel, 255, self.Track_onChange)
            cv2.createTrackbar(self._trackbar_LVchannel, self._trackbar_name, self._LVchannel, 255, self.Track_onChange)
            cv2.createTrackbar(self._trackbar_HVchannel, self._trackbar_name, self._HVchannel, 255, self.Track_onChange)

            #cv2.createTrackbar(self._canny_high, self._trackbar_name, self._high, 1000, self.Track_onChange)
            #cv2.createTrackbar(self._canny_low, self._trackbar_name, self._low, 1000, self.Track_onChange)

        elif keycode == 110: # n
            pass

        elif keycode == 109: # m 
            pass



if __name__=="__main__":
    project4().run()