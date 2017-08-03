import cv2
import numpy as np
from managers import WindowManager, CaptureManager
import scipy
import matplotlib.pyplot as plt
import os



class project2(object):
    
    def __init__(self):
        # window
        self._Window_name = 'project2'
        self._trackbar_Schannel = "Schannel"
        self._trackbar_Vchannel = "Vchannel"

        # image 
        self._Schannel = 100
        self._Vchannel = 254

        # data sign
        self._number_string = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self._MINAREA = 230.0
        self._contour_number = 0
        self._color_sample = []
        self._centeral_coordinate = []

        # k - means
        self._cluster_number = 4
        self._maxDist = 100.0
        self._minDist = 100000.0
        self._clusterAssment = np.zeros([len(self._color_sample), 1])

        # show 
        self._text_flag = False
        self._show_flag = False
        self._Kmeans_flag = False
        self._number_flag = False
        self._count = 0

        # import init
        self._windowManager = WindowManager(self._Window_name, self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, False)



    def Track_onChange(self, back):
        """change value"""

        self._Schannel = cv2.getTrackbarPos(self._trackbar_Schannel, self._Window_name)
        self._Vchannel = cv2.getTrackbarPos(self._trackbar_Vchannel, self._Window_name)



    def color_catch(self, img, cy, cx, room, step):
        """get (b,g,r) color set"""

        room_step_x = int(room / step)
        room_step_y = int(room / step)
        
        cx = cx - int(room / 2)
        cy = cy - int(room / 2)


        if (cx + room) >= 470 or (cx - room) <= 0 :
            room_step_x = 1


        if (cy + room) >= 630 or (cy - room) <= 0:
            room_step_y = 1


        color_temp_b = np.zeros([room_step_x, room_step_y])
        color_temp_g = np.zeros([room_step_x, room_step_y])
        color_temp_r = np.zeros([room_step_x, room_step_y])

        for x in range(room_step_x):
            for y in range(room_step_y):
                color_temp_b = img.item(cx+(x*step), cy+(y*step), 0)
                color_temp_g = img.item(cx+(x*step), cy+(y*step), 1)
                color_temp_r = img.item(cx+(x*step), cy+(y*step), 2)

        return int(np.mean(color_temp_b)), int(np.mean(color_temp_g)), int(np.mean(color_temp_r))



    def euclDistance(self,vector1, vector2): 
        """calculate distance : vector2 - vector1"""
        return np.sqrt(np.sum(np.power((vector2 - vector1), 2)))



    def k_means(self, sample, cluster_count):
        """K_means . need sample and cluster_count"""
        kmeans_flag = False

        # step 1 : get samples and sample number : N 
        sample_len = len(sample)
        sample_dim = len(sample[0])

        # step 2 : get init cluster,named centroids
        centroids = np.zeros([cluster_count, sample_dim])
        sample = np.array(sample)
        sample4chose = sample
        
        # get first random centroids
        random_number = np.random.randint(0, sample_len, 1)
        centroids[0] = sample4chose[random_number]
        sample4chose = np.delete(sample4chose, random_number, 0)

        # get other centroids
        if cluster_count > 1:
            for cnt in range(cluster_count - 1):
                maxDist = self._maxDist
                maxIndex = 0
                for i in range(len(sample4chose)):
                    distance = 0
                    for k in range(cnt+1):
                        distance += self.euclDistance(centroids[k, :], sample4chose[i, :])
                    if distance > maxDist:
                        maxDist = distance
                        maxIndex = i
                centroids[cnt+1] = sample4chose[maxIndex]
                sample4chose = np.delete(sample4chose, maxIndex, 0)

        # get other parameter
        clusterAssment = np.zeros([sample_len, 1])
        clusterChanged = True

        # step 3 : K-means loop 
        while clusterChanged:
            clusterChanged = False
        
            # each sample
            for i in range(sample_len):
                minDist = self._minDist
                minIndex = 0

                # find the centroid who is closest
                for j in range(cluster_count):
                    distance = self.euclDistance(centroids[j, :], sample[i, :])
                    if distance < minDist:
                        minDist = distance
                        minIndex = j

                # update
                if clusterAssment[i, 0] != minIndex:
                    clusterAssment[i, 0] = minIndex
                    clusterChanged = True

            #reset centroids
            centroids = np.zeros([cluster_count, sample_dim])

            # get avg centroids number
            for j in range(cluster_count):
                # create temp array
                centroids_temp = np.zeros([1,sample_dim])

                # load cluster occur array
                for i in range(sample_len):
                    if clusterAssment[i] == j:
                        centroids_temp = np.row_stack((centroids_temp, sample[i]))

                # calculate avg 
                centroids_temp = np.delete(centroids_temp, 0, 0)
                centroids[j] = np.mean(centroids_temp, 0)

        kmeans_flag = True
        return kmeans_flag, centroids, clusterAssment



    def auto_k_means(self, sample):
        """K_means . need sample and cluster_count"""
        kmeans_flag = False
        auto_flag = True
        sample = np.array(sample)
        cluster_count = 0
        siMAX = 0
        cluster_count_right = 1

        # auto K-means loop
        while(auto_flag):
            cluster_count += 1
            if cluster_count == len(sample):
                auto_flag = False

            # get clusterAssment
            kmeans_flag, centroids, clusterAssment = self.k_means(sample, cluster_count)

            # init three parameter
            ai = np.zeros([len(sample), 1])
            bi = np.zeros([len(sample), 1])
            si = np.zeros([len(sample), 1])

            # calculate s 
            for i in range(len(sample)):

                # ai
                dist_ai = np.zeros([1,1])
                mean_flag = False
                for j in range(len(sample)):
                    if (clusterAssment[i] == clusterAssment[j]) and (i != j):
                        mean_flag = True
                        dist_ai = np.row_stack((dist_ai, np.array(self.euclDistance(sample[j], sample[i]))))

                if mean_flag:
                    dist_ai = np.delete(dist_ai, 0, 0)
                    ai[i] = np.mean(dist_ai, 0)


                # bi
                biMIN = 10000.0
                for k in range(cluster_count):
                    if clusterAssment[i] != k:
                        dist_bi = np.zeros([1,1])
                        for j in range(len(sample)):
                            if clusterAssment[j] == k:
                                dist_bi = np.row_stack((dist_bi, np.array(self.euclDistance(sample[j], sample[i]))))
                            
                        dist_bi = np.delete(dist_bi, 0, 0)
                        bi_avg = np.mean(dist_bi, 0)

                        if bi_avg < biMIN:
                            bi[i] = bi_avg
                            biMIN = bi_avg

                        #if self._show_flag:
                        #    print('\nbi_avg : ', bi_avg)

                if ai[i] > bi[i]:
                    si[i] = (bi[i] - ai[i])/ai[i]

                elif ai[i] < bi[i]:
                    si[i] = (bi[i] - ai[i])/bi[i]

                else :
                    si[i] = 0
            
            s = np.mean(si, 0)
            if s > siMAX:
                cluster_count_right = cluster_count
                siMAX = s

            if self._show_flag:
                #print('\nai : ', ai)
                #print('\nbi : ', bi)
                print('\nsample : ', sample)


        #if self._show_flag:
        #    #print('\nbi : ', bi)
        #    print('\ncluster_count_right : ', cluster_count_right)


        kmeans_flag, centroids, clusterAssment = self.k_means(sample, cluster_count_right)
        
        self._show_flag = False

        return kmeans_flag, centroids, clusterAssment



    def camdis(self, frame):
        # load picture
        srcRGB = frame

        # convert to hsv and split it
        hsv = cv2.cvtColor(srcRGB, cv2.COLOR_BGR2HSV)
        (img_h, img_s, img_v) = cv2.split(hsv)

        # H channel (Thresh_img1)
        Thresh_img1 = img_h

        # S channel (Thresh_img2)
        Thresh_img2 = img_s
        retval, Thresh_img2 = cv2.threshold(Thresh_img2, self._Schannel, 255, cv2.THRESH_BINARY) #3x3 73

        kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        Thresh_img2 = cv2.morphologyEx(Thresh_img2, cv2.MORPH_OPEN, kernel_3x3)
        Thresh_img2 = cv2.morphologyEx(Thresh_img2, cv2.MORPH_CLOSE, kernel_3x3)

        # V channel (Thresh_img3)
        Thresh_img3 = img_v
        retval, Thresh_img3 = cv2.threshold(Thresh_img3, self._Vchannel, 255, cv2.THRESH_BINARY) #no equalizeHist 245

        # add v channel to s channel
        Thresh_img = cv2.add(Thresh_img3, Thresh_img2)

        # color detect
        srcHSV = cv2.merge((img_h, img_s, img_v))
        srcHSV = cv2.morphologyEx(srcHSV, cv2.MORPH_OPEN, kernel_5x5)
        srcHSV = cv2.morphologyEx(srcHSV, cv2.MORPH_CLOSE, kernel_5x5)

        # seek targets
        Thresh_img, contours, hier = cv2.findContours(Thresh_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # init for show data in window
        self._contour_number = 0
        self._centeral_coordinate = []
        self._color_sample = []

        # sign
        for cnt in contours:

            #move area  that less than set value
            Length = cv2.arcLength(cnt, 0)
            if Length < self._MINAREA:
                continue

            if Length >= self._MINAREA:
                #count contour number
                self._contour_number += 1

                # draw the contours
                cv2.drawContours(srcRGB, cnt, -1, (255, 255, 0), 3)

                # seek center of the image
                M = cv2.moments(cnt)
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])

                # get color value 
                #if (40< cY and cY < 440) or (40< cX and cX < 600) :
                (b, g, r) = self.color_catch(srcRGB, cX, cY, 40, 2)
                self._color_sample.append((b, g, r))

                # draw and record the center
                self._centeral_coordinate.append((cX, cY))
                        
        # show image
        #cv2.imshow('src', src)
        #cv2.imshow('Thresh_img', Thresh_img)
        #cv2.imshow('Thresh_img1', Thresh_img1)
        #cv2.imshow('Thresh_img2', Thresh_img2)
        #cv2.imshow('Thresh_img3', Thresh_img3)

        if self._Kmeans_flag:
            # use K-means to find color set
            if self._color_sample :

                #self._text_flag, centroids, clusterAssment = self.auto_k_means(self._color_sample)
                self._text_flag, centroids, self._clusterAssment = self.k_means(self._color_sample, self._cluster_number)

            self._Kmeans_flag = False

        # draw color set
        if self._text_flag:
            self._clusterAssment = np.array(self._clusterAssment)
            for cnt in range(self._contour_number):
                (cX,cY) = self._centeral_coordinate[cnt]
                cv2.circle(srcRGB, (cX, cY), 5, (0, 0, 0), -1)
                cv2.putText(srcRGB, self._number_string[int(self._clusterAssment[cnt])], (cX - 10, cY - 10),
		                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if self._number_flag:
            for cnt in range(self._contour_number):
                (cX,cY) = self._centeral_coordinate[cnt]
                cv2.circle(srcRGB, (cX, cY), 5, (0, 0, 0), -1)
                cv2.putText(srcRGB, self._number_string[cnt], (cX - 10, cY + 30),
		                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)



    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            
            if frame is not None:
                self.camdis(frame)

                ## load picture
                #srcRGB = frame

                ## convert to hsv and split it
                #hsv = cv2.cvtColor(srcRGB, cv2.COLOR_BGR2HSV)
                #(img_h, img_s, img_v) = cv2.split(hsv)

                ## H channel (Thresh_img1)
                #Thresh_img1 = img_h

                ## S channel (Thresh_img2)
                #Thresh_img2 = img_s
                #retval, Thresh_img2 = cv2.threshold(Thresh_img2, self._Schannel, 255, cv2.THRESH_BINARY) #3x3 73

                #kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
                #kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
                #Thresh_img2 = cv2.morphologyEx(Thresh_img2, cv2.MORPH_OPEN, kernel_3x3)
                #Thresh_img2 = cv2.morphologyEx(Thresh_img2, cv2.MORPH_CLOSE, kernel_3x3)

                ## V channel (Thresh_img3)
                #Thresh_img3 = img_v
                #retval, Thresh_img3 = cv2.threshold(Thresh_img3, self._Vchannel, 255, cv2.THRESH_BINARY) #no equalizeHist 245

                #img = frame

                #kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
                #kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
                #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_5x5)
                #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_5x5)

                #img = cv2.Canny(img,150,600)


                ## add v channel to s channel
                #Thresh_img = Thresh_img2 - img


                #Thresh_img, contours, hier = cv2.findContours(Thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                #for cnt in contours:

                #    #move area  that less than set value
                #    Length = cv2.arcLength(cnt, 0)
                #    if Length < self._MINAREA:
                #        continue

                #    if Length >= self._MINAREA:
                #        #count contour number
                #        self._contour_number += 1

                #        # draw the contours
                #        #cv2.drawContours(frame, cnt, -1, (255, 255, 0), 3)

                #cv2.imshow('canny', Thresh_img)

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
            for i in range(self._contour_number):
                print('\nThe',(i+1) , 'object :')
                print('\ncenteral coordinate :', self._centeral_coordinate[i])
                
        elif keycode == 111: # o
            self._show_flag = True
            self._Kmeans_flag = True

        elif keycode == 99: # c

            self._text_flag = False
            self._number_flag = False

        elif keycode == 110: # n
            self._count += 1
            if self._count%2 == 1:
                self._number_flag = True

            else :                
                self._number_flag = False

            if self._count == 10:
                self._count = 0

        elif keycode == 109: # m 
            max_num = 255
            Schannel_value = 100
            Vchannel_value = 254

            cv2.createTrackbar(self._trackbar_Schannel, self._Window_name, Schannel_value, max_num, self.Track_onChange)
            cv2.createTrackbar(self._trackbar_Vchannel, self._Window_name, Vchannel_value, max_num, self.Track_onChange)



if __name__=="__main__":
    project2().run()












# program 

    #def auto_k_means(self, sample):
    #    """K_means . need sample and cluster_count"""
    #    accurate_cluster = np.zeros([10,1])
    #    accurate_flag = 0

    #    while(accurate_flag != 10):
    #        accurate_flag += 1

    #        kmeans_flag = False
    #        auto_flag = True

    #        sample = np.array(sample)
    #        R4compare = np.zeros([len(sample),1])
    #        cluster_count = 0
    #        self._R4compare = 100000.0
    #        temp = 1000.0

    #        # auto K-means loop
    #        while(auto_flag):
    #            cluster_count += 1
    #            if cluster_count == len(sample):
    #                auto_flag = False

    #            kmeans_flag, centroids, clusterAssment = self.k_means(sample, cluster_count)

    #            # calculate average dispersion
    #            avg_dispersion = np.zeros([cluster_count,1])
    #            R = np.zeros([cluster_count,1])

    #            for j in range(cluster_count):
    #                dispersion_dist = 0
    #                occur = 0
    #                for i in range(len(sample)):
    #                    if clusterAssment[i] == j:
    #                        occur += 1
    #                        dispersion_dist += self.euclDistance(centroids[j, :], sample[i, :])
    #                avg_dispersion[j] = dispersion_dist / occur

    #            # calculate DB
    #            R = np.zeros([cluster_count, 1])
    #            for j in range(cluster_count):
    #                for k in range(cluster_count):
    #                    if k == j or self.euclDistance(centroids[j], centroids[k]) == 0 :
    #                        continue
   
    #                    R_temp = (avg_dispersion[j] + avg_dispersion[k]) / self.euclDistance(centroids[j], centroids[k])

    #                    if R[j] < R_temp:
    #                        R[j] = R_temp

    #            R4compare_temp = np.mean(R, 0)

    #            if abs(R4compare_temp - temp) < self._R4compare :
    #                self._R4compare = abs(R4compare_temp - temp)

    #                cluster_count_right = cluster_count - 1 

    #                temp = R4compare_temp

    #        accurate_cluster[accurate_flag - 1] = cluster_count_right

    #    cluster_count_right = np.mean(accurate_cluster)

    #    if self._show_flag:
    #        print('\naccurate_cluster : ', accurate_cluster)
    #        print('\ncluster_count_right : ', cluster_count_right)


    #    kmeans_flag, centroids, clusterAssment = self.k_means(sample, int(cluster_count_right))
        
    #    self._show_flag = False

    #    return kmeans_flag, centroids, clusterAssment