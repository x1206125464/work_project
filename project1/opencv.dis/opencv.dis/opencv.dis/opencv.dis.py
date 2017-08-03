import cv2
import numpy as np


class project1(object):

    def __init__(self, img):
        self._MINAREA = 200
        self._img = img
        self._Schannel_threshold = 112        #Source 76    #Source1 112   #Source2 98
        self._Vchannel_threshold = 254        #Source 245   #Source1 254   #Source2 254
        self._cluster_number = 4
        self._maxDist = 100.0
        self._minDist = 100000.0


    def color_catch(self, img, cx, cy, room):
        """get (b,g,r) color set"""
        x_b = 0
        x_g = 0
        x_r = 0

        y_b = 0
        y_g = 0
        y_r = 0

        for x in range(room):
            for y in range(room):
                y_b += img.item(cy+x, cx+y, 0)
                y_g += img.item(cy+x, cx+y, 1)
                y_r += img.item(cy+x, cx+y, 2)

            y_b = int(y_b / room)
            y_g = int(y_g / room)
            y_r = int(y_r / room)

            x_b += y_b
            x_g += y_g
            x_r += y_r

        x_b = int(x_b / room)
        x_g = int(x_g / room)
        x_r = int(x_r / room)

        return x_b, x_g, x_r


    def euclDistance(self,vector1, vector2): 
        """calculate distance : vector2 - vector1"""
        return np.sqrt(np.sum(np.power((vector2 - vector1), 2)))


    def k_means(self, sample, cluster_count):
        """K_means . need sample and cluster_count"""
        # step 1 : get samples and sample number : N 
        sample_len = len(sample)
        sample_dim = len(sample[0])


        # step 2 : get init cluster,named centroids
        centroids = np.zeros([cluster_count, sample_dim])
        sample = np.array(sample)

        # get first random centroids
        random_number = np.random.randint(0, sample_len, 1)
        centroids[0] = sample[random_number]


        # get other centroids
        for cnt in range(cluster_count - 1):
            maxDist = self._maxDist
            maxIndex = 0
            for i in range(sample_len):
                distance = 0
                for k in range(cnt+1):
                    distance += self.euclDistance(centroids[k, :], sample[i, :])
                if distance > maxDist:
                    maxDist = distance
                    maxIndex = i
            centroids[cnt+1] = sample[maxIndex]


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

            #recalculate centroids avg
            centroids = np.zeros([cluster_count, sample_dim])

            # get occur nunber
            for j in range(cluster_count):
                occur_number = 0
                avg_number = 0
                for i in range(sample_len):
                    if clusterAssment[i] == j:
                        occur_number += 1
            
                # create temp array
                centroids_temp = np.zeros([occur_number, sample_dim])

                # reload and recalculate
                for i in range(sample_len):
                    if clusterAssment[i] == j:
                        avg_number += 1
                        centroids_temp[avg_number - 1] = sample[i]
                        centroids[j] = np.mean(centroids_temp, 0)

            #    print('centroids_temp : \n',centroids_temp)

        #print('sample : \n',sample)
        #print('sample : ',centroids_temp)
        #print('sample : \n',clusterAssment)
        return centroids, clusterAssment


    def Run(self):
        """Run the main loop"""

        # load picture
        srcRGB = cv2.imread(self._img)
        src = cv2.cvtColor(srcRGB, cv2.COLOR_BGR2GRAY)


        # convert to hsv and split it
        hsv = cv2.cvtColor(srcRGB, cv2.COLOR_BGR2HSV)
        (img_h, img_s, img_v) = cv2.split(hsv)


        # H channel (Thresh_img1)
        Thresh_img1 = img_h


        # S channel (Thresh_img2)
        retval, Thresh_img2 = cv2.threshold(img_s.copy(), self._Schannel_threshold, 255, cv2.THRESH_BINARY) #3x3 73

        #kernel_3x3 = np.array([[-1, -1, -1],
        #                       [-1,  8, -1],
        #                       [-1, -1, -1]])

        #kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
        #                       [-1,  1,  2,  1, -1],
        #                       [-1,  2,  4,  2, -1],
        #                       [-1,  1,  2,  1, -1],
        #                       [-1, -1, -1, -1, -1]])

        kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))

        #Thresh_img2 = cv2.erode(Thresh_img2, kernel_5x5)
        #Thresh_img2 = cv2.dilate(Thresh_img2, kernel_5x5)

        Thresh_img2 = cv2.morphologyEx(Thresh_img2, cv2.MORPH_OPEN, kernel_3x3)
        Thresh_img2 = cv2.morphologyEx(Thresh_img2, cv2.MORPH_CLOSE, kernel_3x3)


        # V channel (Thresh_img3)
        Thresh_img3 = img_v
        retval, Thresh_img3 = cv2.threshold(Thresh_img3, self._Vchannel_threshold, 255, cv2.THRESH_BINARY) #no equalizeHist 245
        #img_h = cv2.equalizeHist(img_h)


        # add v channel to s channel
        Thresh_img = cv2.add(Thresh_img3, Thresh_img2)


        # color detect
        srcHSV = cv2.merge((img_h, img_s, img_v))
        srcHSV = cv2.morphologyEx(srcHSV, cv2.MORPH_OPEN, kernel_5x5)
        srcHSV = cv2.morphologyEx(srcHSV, cv2.MORPH_CLOSE, kernel_5x5)


        # seek targets
        Thresh_img, contours, hier = cv2.findContours(Thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        # draw the contours
        Thresh_img_copy = Thresh_img.copy()
        cv2.drawContours(srcRGB, contours, -1, (255, 255, 0), 3)


        # sign
        contour_number = 0
        color_number = 0
        color_sample = []
        centeral_coordinate = []
        number_string = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

        for cnt in contours:
            #move area  that less than set value
            Length = cv2.arcLength(cnt,True)
            if Length < self._MINAREA:
                continue

            if Length >= self._MINAREA:
                # count contour number
                contour_number += 1
                print('\nThe contour number :', contour_number)

                # seek center of the image
                M = cv2.moments(cnt)
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])

                # draw and record the center
                print('Central coordinate of target :', (cY, cX))
                centeral_coordinate.append((cX, cY))
                cv2.circle(srcRGB, (cX, cY), 5, (0, 0, 0), -1)

                # get color value 
                (b, g, r) = self.color_catch(srcRGB, cX, cY, 20)
                color_sample.append((b, g, r))
                print('color :', (b, g, r))


        # use K-means to find color set
        centroids, clusterAssment = self.k_means(color_sample, self._cluster_number)


        # draw color set
        clusterAssment = np.array(clusterAssment)
        for cnt in range(contour_number):
            (cX,cY) = centeral_coordinate[cnt]
            cv2.putText(srcRGB, number_string[int(clusterAssment[cnt])], (cX - 10, cY - 10),
		                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        #print('centroids:', centroids)
        #print('clusterAssment:', clusterAssment)
        #cv2.imshow('img_h', img_h)
        #cv2.imshow('img_s', img_s)
        #cv2.imshow('img_v', img_v)
        #cv2.imshow('Thresh_img1', Thresh_img1)
        #cv2.imshow('Thresh_img2', Thresh_img2)
        #cv2.imshow('Thresh_img3', Thresh_img3)
        #cv2.imshow('Thresh_img', Thresh_img)
        #cv2.imshow('Thresh_img_copy', Thresh_img_copy)
        #cv2.imshow('targets_image', Thresh_img)
        #cv2.imshow('srcHSV', srcHSV)
        #cv2.imshow('hsv', hsv)
        cv2.imshow('srcRGB', srcRGB)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__=="__main__":
    project1('Source1.jpg').Run()



