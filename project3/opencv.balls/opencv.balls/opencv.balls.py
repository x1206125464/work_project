import cv2
import numpy as np
from managers import WindowManager, CaptureManager
import scipy
import matplotlib.pyplot as plt
import os



class project3(object):
    
    def __init__(self):
        # window
        self._Window_name = 'project2'
        self._trackbar_Schannel = "Schannel"
        self._trackbar_Vchannel = "Vchannel"

        # coordinate 
        self._M_record = np.zeros([1,2])
        self._contour_number = 0

        # image 
        self._Schannel = 100
        self._Vchannel = 254

        self._Rchannel = 100
        self._Gchannel = 100
        self._Bchannel = 100

        # data sign
        self._MINAREA = 300.0

        # show 
        self._show_flag = False
        self._test_flag = False
        self._count = 0

        # import init
        self._windowManager = WindowManager(self._Window_name, self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, False)



    def Track_onChange(self, back):
        """change value"""

        self._Schannel = cv2.getTrackbarPos(self._trackbar_Schannel, self._Window_name)
        self._Vchannel = cv2.getTrackbarPos(self._trackbar_Vchannel, self._Window_name)
        self._Rchannel = cv2.getTrackbarPos(self._trackbar_Schannel, self._Window_name)
        self._Gchannel = cv2.getTrackbarPos(self._trackbar_Schannel, self._Window_name)
        self._Bchannel = cv2.getTrackbarPos(self._trackbar_Schannel, self._Window_name)



    def euclDistance(self,vector1, vector2): 
        """calculate distance : vector2 - vector1"""
        return np.sqrt(np.sum(np.power((vector2 - vector1), 2)))



    def filter(self, src, low, high, ksize_open, ksize_close, flag):
        lowHSV = np.array(low)
        highHSV = np.array(high)
        src_Filter = cv2.inRange(src, lowHSV, highHSV)
        src_Filter = cv2.medianBlur(src_Filter, 9)

        if flag == True:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(ksize_open, ksize_open))
            src_Filter = cv2.morphologyEx(src_Filter, cv2.MORPH_OPEN, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(ksize_close, ksize_close))
            src_Filter = cv2.morphologyEx(src_Filter, cv2.MORPH_CLOSE, kernel)

        return src_Filter

        

    def campick(self, frame):
        # load picture
        srcRGB = frame

        # convert to hsv and equalize it
        hsv = cv2.cvtColor(srcRGB, cv2.COLOR_BGR2HSV)
        (img_h, img_s, img_v) = cv2.split(hsv)
        img_v = cv2.equalizeHist(img_v)
        hsv = cv2.merge((img_h, img_s, img_v))

        # blue image 
        lowHSV = [107, 30, 0]
        highHSV = [150, 100, 110]

        Thresh_img1 = self.filter(hsv, lowHSV, highHSV, 13, 9, True)
        Thresh_img1, contours1, hier = cv2.findContours(Thresh_img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(srcRGB, contours1, -1, (255, 0, 0), -1)

        contours_temp = np.zeros([1, 1, 2], np.int)
        for cnt in contours1:
            contours_temp = np.row_stack((contours_temp, np.array(cnt)))
        contours_temp = np.delete(contours_temp, [0, 0, 2], 0)
        rect1 = cv2.minAreaRect(contours_temp)
        box1 = np.int0(cv2.boxPoints(rect1)) 
        #cv2.drawContours(srcRGB, [box1], 0, (0,0,0), 3)

        # green image 
        lowHSV = [54, 60, 28]
        highHSV = [103, 255, 140]

        Thresh_img2 = self.filter(hsv, lowHSV, highHSV, 13, 9, True)
        Thresh_img2, contours2, hier = cv2.findContours(Thresh_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(srcRGB, contours2, -1, (0, 255, 0), -1)

        contours_temp = np.zeros([1, 1, 2], np.int)
        for cnt in contours2:
            contours_temp = np.row_stack((contours_temp, np.array(cnt)))
        contours_temp = np.delete(contours_temp, [0, 0, 2], 0)
        rect2 = cv2.minAreaRect(contours_temp)
        box2 = np.int0(cv2.boxPoints(rect2))
        #cv2.drawContours(srcRGB, [box2], 0, (0,0,0), 3)

        # red image 
        lowHSV = [154, 40, 103]
        highHSV = [179, 255, 255]
        Thresh_img3_1 = self.filter(hsv, lowHSV, highHSV, 1, 15, True)

        lowHSV = [0, 143, 80]
        highHSV = [10, 255, 255]
        Thresh_img3_2 = self.filter(hsv, lowHSV, highHSV, 1, 15, True)

        Thresh_img3 = Thresh_img3_1 + Thresh_img3_2
        Thresh_img3, contours3, hier = cv2.findContours(Thresh_img3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(srcRGB, contours3, -1, (0, 0, 255), -1)

        contours_temp = np.zeros([1, 1, 2], np.int)
        for cnt in contours3:
            contours_temp = np.row_stack((contours_temp, np.array(cnt)))
        contours_temp = np.delete(contours_temp, [0, 0, 2], 0)
        rect3 = cv2.minAreaRect(contours_temp)
        box3 = np.int0(cv2.boxPoints(rect3))
        #cv2.drawContours(srcRGB, [box3], 0, (0,0,0), 3)

        # find object
        Thresh_img4_temp = Thresh_img1 + Thresh_img2 + Thresh_img3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(43, 43))
        Thresh_img4 = cv2.morphologyEx(Thresh_img4_temp, cv2.MORPH_CLOSE, kernel)
        mask = np.zeros([482,642], np.uint8)
        retval, object_img, mask, rect = cv2.floodFill(Thresh_img4.copy(), mask, (0,0), 255)
        object_img = 255 - object_img
        object_img, contours, hier = cv2.findContours(object_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            Length = cv2.arcLength(cnt, 0)
            if Length < self._MINAREA:
                continue

            if Length >= self._MINAREA:
                rect = cv2.minAreaRect(cnt)
                box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(srcRGB, [box], 0, (0,0,0), 3)

                M = cv2.moments(cnt)
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])

                if abs(self._M_record[0,0] - cX) + abs(self._M_record[0,1] - cY) <= 4 and self._show_flag == False :
                    retval1 = cv2.pointPolygonTest(box1, (cX, cY), 0)               
                    retval2 = cv2.pointPolygonTest(box2, (cX, cY), 0)                
                    retval3 = cv2.pointPolygonTest(box3, (cX, cY), 0)                
                    if retval1 == 1 and retval2 == -1 and retval3 == -1:
                        print('\nblue :', retval1)
                        print('\ncenteral coordinate :', [cY, cX])
                        self._show_flag = True

                    elif retval1 == -1 and retval2 == 1 and retval3 == -1:
                        print('\ngreen :', retval2)
                        print('\ncenteral coordinate :', [cY, cX])
                        self._show_flag = True

                    elif retval1 == -1 and retval2 == -1 and retval3 == 1:
                        print('\nred :', retval3)
                        print('\ncenteral coordinate :', [cY, cX])
                        self._show_flag = True

                    elif retval1 == -1 and retval2 == 1 and retval3 == 1:
                        print('\ngreen :', retval2)
                        print('\ncenteral coordinate :', [cY, cX])
                        self._show_flag = True

                    elif retval1 == 1 and retval2 == 1 and retval3 == -1:
                        print('\nblue :', retval1)
                        print('\ncenteral coordinate :', [cY, cX])
                        self._show_flag = True

                    else:
                        pass

                self._M_record[0,0] = cX
                self._M_record[0,1] = cY
                cv2.circle(srcRGB, (cX, cY), 5, (0, 0, 0), -1)





    def camtest(self, frame):
        # load picture
        srcRGB = frame

        # create ROI mask
        mask_ROI = np.zeros([480,640], np.uint8)

        # convert to hsv and equalize it
        hsv = cv2.cvtColor(srcRGB, cv2.COLOR_BGR2HSV)
        (img_h, img_s, img_v) = cv2.split(hsv)
        img_v = cv2.equalizeHist(img_v)
        hsv = cv2.merge((img_h, img_s, img_v))

        # catch blue image and set blue ROI
        lowHSV = [107, 30, 0]
        highHSV = [150, 100, 110]
        Thresh_img1 = self.filter(hsv, lowHSV, highHSV, 13, 9, True)
        Thresh_img1, contours1, hier = cv2.findContours(Thresh_img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_temp = np.zeros([1, 1, 2], np.int)
        for cnt in contours1:
            contours_temp = np.row_stack((contours_temp, np.array(cnt)))
        contours_temp = np.delete(contours_temp, [0, 0, 2], 0)
        rect1 = cv2.minAreaRect(contours_temp)
        box1 = np.int0(cv2.boxPoints(rect1)) 
        cv2.drawContours(mask_ROI, [box1], 0, (255,255,255), 5)         # set blue ROI on mask
        #cv2.drawContours(srcRGB, contours1, -1, (255, 0, 0), -1)       # blue cover 
        #cv2.drawContours(srcRGB, [box1], 0, (0,0,0), 3)                # draw mini area on frame

        # green image 
        lowHSV = [54, 60, 28]
        highHSV = [103, 255, 140]
        Thresh_img2 = self.filter(hsv, lowHSV, highHSV, 13, 9, True)
        Thresh_img2, contours2, hier = cv2.findContours(Thresh_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_temp = np.zeros([1, 1, 2], np.int)
        for cnt in contours2:
            contours_temp = np.row_stack((contours_temp, np.array(cnt)))
        contours_temp = np.delete(contours_temp, [0, 0, 2], 0)
        rect2 = cv2.minAreaRect(contours_temp)
        box2 = np.int0(cv2.boxPoints(rect2))
        cv2.drawContours(mask_ROI, [box2], 0, (255,255,255), 5)         # set green ROI on mask
        #cv2.drawContours(srcRGB, contours2, -1, (0, 255, 0), -1)       # green cover 
        #cv2.drawContours(srcRGB, [box2], 0, (0,0,0), 3)                # draw mini area on frame

        # red image 
        lowHSV = [154, 40, 103]
        highHSV = [179, 255, 255]
        Thresh_img3_1 = self.filter(hsv, lowHSV, highHSV, 1, 15, True)
        lowHSV = [0, 143, 80]
        highHSV = [10, 255, 255]
        Thresh_img3_2 = self.filter(hsv, lowHSV, highHSV, 1, 15, True)
        Thresh_img3 = Thresh_img3_1 + Thresh_img3_2
        Thresh_img3, contours3, hier = cv2.findContours(Thresh_img3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_temp = np.zeros([1, 1, 2], np.int)
        for cnt in contours3:
            contours_temp = np.row_stack((contours_temp, np.array(cnt)))
        contours_temp = np.delete(contours_temp, [0, 0, 2], 0)
        rect3 = cv2.minAreaRect(contours_temp)
        box3 = np.int0(cv2.boxPoints(rect3))
        cv2.drawContours(mask_ROI, [box3], 0, (255,255,255), 5)         # set red ROI on mask
        #cv2.drawContours(srcRGB, contours3, -1, (0, 0, 255), -1)       # red cover 
        #cv2.drawContours(srcRGB, [box3], 0, (0,0,0), 3)                # draw mini area on frame

        # find object
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(41, 41))
        Thresh_img4_temp = Thresh_img1 + Thresh_img2 + Thresh_img3
        mask_ROI = cv2.morphologyEx(mask_ROI, cv2.MORPH_CLOSE, kernel)
        mask_ROI, contours_ROI, hier = cv2.findContours(mask_ROI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(Thresh_img4_temp, contours_ROI, -1, (255, 255, 255), 3)
        Thresh_img4 = cv2.morphologyEx(Thresh_img4_temp, cv2.MORPH_CLOSE, kernel)
        mask = np.zeros([482,642], np.uint8)
        retval, object_img, mask, rect = cv2.floodFill(Thresh_img4.copy(), mask, (0,0), 255)
        object_img = 255 - object_img
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25, 25))
        #object_img = cv2.morphologyEx(object_img, cv2.MORPH_OPEN, kernel)
        object_img, contours, hier = cv2.findContours(object_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # tagged object
        for cnt in contours:
            Length = cv2.arcLength(cnt, 0)
            if Length < self._MINAREA:
                continue

            if Length >= self._MINAREA:
                M = cv2.moments(cnt)
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                rect = cv2.minAreaRect(cnt)
                box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(srcRGB, [box], 0, (0,0,0), 3)
                if abs(self._M_record[0,0] - cX) + abs(self._M_record[0,1] - cY) <= 4 and self._show_flag == False :
                    retval1 = cv2.pointPolygonTest(box1, (cX, cY), 0)               
                    retval2 = cv2.pointPolygonTest(box2, (cX, cY), 0)                
                    retval3 = cv2.pointPolygonTest(box3, (cX, cY), 0)        
                    if retval1 == -1 and retval2 == -1 and retval3 == -1:      
                        cX = cX - 5
                        retval1 = cv2.pointPolygonTest(box1, (cX, cY), 0)               
                        retval2 = cv2.pointPolygonTest(box2, (cX, cY), 0)                
                        retval3 = cv2.pointPolygonTest(box3, (cX, cY), 0)
                        cX = cX + 5

                    if retval1 == 1 and retval2 == -1 and retval3 == -1:
                        print('\nblue :', retval1)
                        print('\ncenteral coordinate :', [cY, cX])
                        self._show_flag = True

                    elif retval1 == -1 and retval2 == 1 and retval3 == -1:
                        print('\ngreen :', retval2)
                        print('\ncenteral coordinate :', [cY, cX])
                        self._show_flag = True

                    elif retval1 == -1 and retval2 == -1 and retval3 == 1:
                        print('\nred :', retval3)
                        print('\ncenteral coordinate :', [cY, cX])
                        self._show_flag = True

                    elif retval1 == -1 and retval2 == 1 and retval3 == 1:
                        print('\ngreen :', retval2)
                        print('\ncenteral coordinate :', [cY, cX])
                        self._show_flag = True

                    elif retval1 == 1 and retval2 == 1 and retval3 == -1:
                        print('\nblue :', retval1)
                        print('\ncenteral coordinate :', [cY, cX])
                        self._show_flag = True

                    else:                        
                        pass

                self._M_record[0,0] = cX
                self._M_record[0,1] = cY
                cv2.circle(srcRGB, (cX, cY), 5, (0, 0, 0), -1)




        #cv2.imshow('mask_ROI', mask_ROI)
        cv2.imshow('Thresh_img4', Thresh_img4)
        cv2.imshow('Thresh_img4_temp', Thresh_img4_temp)
        cv2.imshow('object_img', object_img)

       

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            
            if frame is not None:
                if self._test_flag == True:
                    self.campick(frame)
                else :
                    self.camtest(frame)

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
            
        elif keycode == 115: # s
            print('\npress s -- show : show data')
            self._show_flag = False
                
        elif keycode == 116: # t
            self._count += 1
            if self._count%2 == 1:
                print('\npress t -- test : switch to test model')
                self._test_flag = True

            else :                
                print('\npress t -- test : switch to work model')
                self._test_flag = False

            if self._count == 10:
                self._count = 0




        elif keycode == 110: # n
            pass

        elif keycode == 109: # m 
            pass





if __name__=="__main__":
    project3().run()