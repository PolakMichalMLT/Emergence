import numpy as np
import pandas as pd
from datetime import datetime
import cv2
import os
from collections import namedtuple 
import pyzbar.pyzbar as pz
import scipy.io
import global_variables as gv

class ImageCropper:

    def __init__(self,filepath, chamber):

        self.filepath = filepath
        self.chamber = chamber


    def name2id(self):

        #try:

        #Get list  of image names in data folder
        file_list = [file for file in os.listdir(self.filepath + 'images/')]

        ids = []
        dates = []
        file_names = [] 

        if self.chamber == 'xyz':

            for i, image in enumerate(file_list):

                meta_split = image.split('.')[0].split('_')
                location = meta_split[4][5:]
                date_meta = meta_split[2][5:].split('-')
                date = datetime(year=int(date_meta[0]), month= int(date_meta[1]), day=int(date_meta[2]), hour=int(date_meta[3]), minute=int(date_meta[4]), second=int(date_meta[5]))

                ids.append(location)    
                dates.append(date)
                file_names.append(image)

        elif self.chamber == 'hydropony':

            for i, image in enumerate(file_list):

                meta_split = image.split('.')[0].split('_')
                location = meta_split[0]
                date_meta = meta_split[1].split('-')
                date = datetime(year=2000+int(date_meta[2]), month= int(date_meta[1]), day=int(date_meta[0]), hour=int(date_meta[3]), minute=int(date_meta[4]), second=int(date_meta[5]))

                ids.append(location)    
                dates.append(date)
                file_names.append(image)


        return ids, dates, file_names

        #except Exception as e: print(e)


    def lensdistort(self, image, k, model_number = 4, bordertype = 'crop'):

        """LENSDISTORT corrects for barrel and pincusion lens abberations
        I = LENSDISTORT(I, k)corrects for radially symmetric distortions, where
        I is the input image and k is the distortion parameter. lens distortion
        can be one of two types: barrel distortion and pincushion distortion.
        In "barrel distortion", image magnification decreases with 
        distance from the optical axis. The apparent effect is that of an image 
        which has been mapped around a sphere (or barrel). In "pincushion 
        distortion", image magnification increases with the distance from the 
        optical axis. The visible effect is that lines that do not go through the 
        centre of the image are bowed inwards, towards the centre of the image, 
        like a pincushion [1]. 
        
        Args:
        
            k: Single numeric value form [-1;1] interval. Parameter of equation, which corrects fishey distortions.
            
            model_number: Single integer value from {1,2,3,4} set. With this argument you choose one from four correction models.
            
            border_type: Parameter for border correction. You can use fit or crop value.
        
        """
        
        #try:
                
        assert (type(image)  == np.ndarray), 'Input data has to be RGB image' 
        assert (len(image.shape) == 3),'Input data has to be RGB image'
        assert np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
        assert (type(k) == float or type(k) == np.float64), 'Distortion parameter has to be float value'
        assert ((k >= -1.0) and (k < 0)) and ((k>0) and (k <= 1.0)), 'Distortion parameter has to be value in [-1;0) U (0;1] interval'
        assert (0 < model_number and model_number < 5), 'You can choose distortion model with value from a set {1,2,3,4}'
        assert (bordertype == 'fit') or (bordertype == 'crop'), 'bordertype argument has to be string, "fit" or "crop"'

        undistorted_image = np.zeros(image.shape)

        #Nested function that pics the model type to be used
        def distortfun(r,k):

            if(model_number==1):

                s = r*(1/(1+k*r))

            elif(model_number==2):

                s = r*(1/(1+k*(r**2)))

            elif(model_number==3):

                s = r*(1+k*r)

            elif(model_number==4):

                s = r*(1+k*(r**2))

            return s


        #Nested function that creates a scaling parameter based on the bordertype selected
        def bordercorrect(r,s,k,center, R):

            if(k < 0):

                if(bordertype == 'fit'):

                    x = s[0]/r[0] 

                elif(bordertype == 'crop'):  

                    x = 1/(1 + k*(min(center)/R)**2)

            elif(k > 0):

                if(bordertype ==  'fit'):

                    x = 1/(1 + k*(min(center)/R)**2)

                elif(bordertype ==  'crop'):

                    x = s[0]/r[0]

            return x



        def imdistcorrect(image,k):

            #Determine the size of the image to be distorted
            h,w = image.shape[0:2]

            center = (w//2,h//2)

            #Creates N x M (#pixels) x-y points
            xi,yi = np.meshgrid(np.arange(w), np.arange(h))

            #Creates converst the mesh into a colum vector of coordiantes relative to the center
            xt = xi - center[0]
            yt = yi - center[1]

            #Function for transforming cartesian coordinates to polar coordinates
            def cart2pol(x, y):
                rho = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                return(rho, theta)

            #Function for transforming polar coordinates to cartesian coordinates
            def pol2cart(rho, theta):
                x = rho * np.cos(theta)
                y = rho * np.sin(theta)
                return(x, y)

            #Converts the x-y coordinates to polar coordinates
            r, theta = cart2pol(xt,yt)

            #Calculate the maximum vector (image center to image corner) to be used
            R = (center[0]**2 + center[1]**2)**(0.5)

            #Normalize the polar coordinate r to range between 0 and 1
            r = r/R

            #Aply the r-based transformation
            s = distortfun(r,k)

            #un-normalize s
            s2 = s * R

            #Find a scaling parameter based on selected border type  
            brcor = bordercorrect(r,s,k, center, R)
            s2 = s2 * brcor

            #Convert back to cartesian coordinates
            ut,vt = pol2cart(s2,theta)

            u = np.reshape(ut,xt.shape) + center[0]
            v = np.reshape(vt,yt.shape) + center[1]

            u = u.astype(np.float32)
            v = v.astype(np.float32)

            #Map for image distortion corrections
            tmap_B = np.dstack([u,v])

            #Remapping
            remapped = cv2.remap(image,tmap_B,None,interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

            return remapped

        undistorted_image = imdistcorrect(image,k)
        
        return undistorted_image
    
        #except Exception as e: print(e)


    def sliding_window(self, image, stepSize, windowSize):

        # Verify correctness of input arguments
        assert (type(image) == np.ndarray), 'Input data has to be RGB image'
        assert ((len(image.shape) == 3) or (
                    len(image.shape) == 2)), 'Input data has to be RGB image or binary image'
        assert np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'

        assert (windowSize[1] < image.shape[1]) & (
                    windowSize[0] < image.shape[0]), 'Window size is bigger than image size'

        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[0], x:x + windowSize[1]])


    def sharp_image(self, image, sigma, power, method):

        """
        Args:

            method: String argument, determines if laplace or kernel sharpening of image is used.

            sigma: Integer argument, valid only for laplace method. It is the size of filter for image median filtering.

            power: Double argument from (0,1) interval, determines power of laplacian for noise reduction.

        """
        # https://www.idtools.com.au/unsharp-masking-python-opencv/

        #try:

        assert (type(method) == str) & ((method == 'kernel') | (
                    method == 'laplace')), 'Method of image sharpening has to be krenle or laplace'
        assert (type(image) == np.ndarray) & (len(image.shape) == 2) & (np.amin(image) >= 0) & (
                    np.amax(image) <= 255), 'Input data has to be grayscale image'
        assert type(
            sigma) == int, 'size gives the shape that is taken from the input array, at every element position, to define the input to the filter function'
        assert ((power > 0) & (power < 1)), 'Value of power parameter has to be in (0,1) interval'

        sharp = np.array()

        if (method == 'laplace'):

            # Median filtering
            image_mf = median_filter(image, sigma)

            # Calculate the Laplacian
            lap = cv2.Laplacian(image_mf, cv2.CV_64F)

            # Calculate the sharpened image
            sharp = image - power * lap

        elif (method == 'kernel'):

            kernel = np.array([[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-2, 2, 2, 2, -1],
                                [-1, -1, -1, -1, -1]]) / 8.0

            # applying the sharpening kernel to the input image & displaying it.
            sharp = cv2.filter2D(image, -1, kernel)

        return sharp

        #except  Exception as e: print(e)


    def segmentation(self, image, low_thresh, up_thresh):

        #try:

        # Verify correctness of input arguments
        assert (type(image) == np.ndarray), 'Input data has to be RGB image'
        assert (len(image.shape) == 3), 'Input data has to be RGB image'
        assert np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'

        assert (type(low_thresh) == np.ndarray) & (len(low_thresh)) & (np.min(low_thresh) >= 0) & (
                    np.max(low_thresh) <= 255), 'Low_thresh input argument is not valid'
        assert (type(up_thresh) == np.ndarray) & (len(up_thresh)) & (np.min(up_thresh) >= 0) & (
                    np.max(up_thresh) <= 255), 'Up_thresh input argument is not valid'

        # Transform RGB image to HSV color space
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # mask for mgreen segmentation
        maskHSV = cv2.inRange(imageHSV, low_thresh, up_thresh)

        # Determine, which pixels are biomass of microgreen
        logic = maskHSV > 0
        # Transform boolean values to integers
        mask = logic.astype('uint8')

        return mask

        #except  Exception as e: print(e)


    def find_qr_code_location(self, image, size, stride, low_thresh, up_thresh):

        """
        find_qr_code_location function provides location of QR code in image. Essential assumption -> one or none qr code object in sector.
        Main idea is remove all noisy (pixels of other objects) pixels and threshold image in a way,that only QR codes pixels are
        visible.

        Args:

            size: Single integer value, which specify size of qr code area/sliding window.

            stride: Single integer value, defines shift of sliding window in pixels.

        """

        #try:

        # Check of input arguments correctness
        assert (type(image) == np.ndarray), 'Input data has to be RGB image'
        assert (len(image.shape) == 3), 'Input data has to be RGB image'
        assert np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'

        assert (type(low_thresh) == np.ndarray) & (len(low_thresh)) & (np.min(low_thresh) >= 0) & (
                    np.max(low_thresh) <= 255), 'Low_thresh input argument is not valid'
        assert (type(up_thresh) == np.ndarray) & (len(up_thresh)) & (np.min(up_thresh) >= 0) & (
                    np.max(up_thresh) <= 255), 'Up_thresh input argument is not valid'

        # image size
        im_height, im_width = image.shape[0:2]

        assert (type(size) == tuple), 'Size of qr code argument is an integer value'

        assert (type(stride) == int), 'Stride has to be integer value, lower than 50'
        assert (stride < 30), 'Stride has to be integer value, lower than 50'

        # Define Rectangle object as namedtuple with four values
        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        # Convenient segmentation of qr codes in image
        segmented = self.segmentation(image, low_thresh, up_thresh)

        # List of x-coordinate of sliding window (top left corner)
        X = []
        # List of y-coordinate of sliding window (top left corner)
        Y = []
        # List of QR code pixels density in sliding window
        dens = []

        # In this for loop, sliding window goes through whole image and computes QR code pixels density
        for (x, y, window) in self.sliding_window(segmented, stepSize=stride, windowSize=(size[0], size[1])):
            if window.shape[0] != size[0] or window.shape[1] != size[1]:
                continue

            value = (window > 0).sum()

            X.append(x)
            Y.append(y)
            dens.append(value)

        # Index of sliding window, which should contain qr code
        index = int(np.argmax(dens))

        # x coordinate of sliding window
        x = X[index]
        # y coordinate of sliding window
        y = Y[index]

        # barcode location
        location = (max(0, x - 5), max(y - 5, 0), min(x + size[1] + 5, im_width), min(y + size[0] + 5, im_height))

        return location

        #except Exception as e: print(e)


    def qr_code_reader(self, image):

        """
        QR_SNAPSHOT_DETECTION function provide information, if some QR code or barcode was detected in sector.
        If some code was detected, it provides decoded data of QR code and it's location.
        That means that in the image is one QR code at maximum. It is essential assumption of this method.
        So in the output you should obtain list with one element. If there is more than one element in a list
        (this can happen only in th first attemp of this method), than raw image was wrongly splitted.

        It uses decode function of pyzbar package and adds some improvements to decode QR code.
        Function try to decode data in several steps. It starts with basic method and in case of failure, it continues with little bit more complicated methods.
        Pyzbar decode function is key element of each method. Each method tries to help with data decoding with some simple improvement.
        You need install pyzbar package on computer, installation is little bit more complicated, but is nicely described here:
        https://www.pyimagesearch.com/2018/05/21/an-opencv-barcode-and-qr-code-scanner-with-zbar/

        Args:

            size: Single integer value, which specify size of qr code area/sliding window.

            stride: Single integer value, defines shift of sliding window in pixels.

            low_thresh: Bottom thresh value for Canny method, (used for qr code area localization in iterative procedure)

            up_thresh: Upper thresh value for Canny method, (used for qr code area localization in iterative procedure)

        """

        #try:
                
        # Check of input arguments correctness
        assert (type(image) == np.ndarray), 'Input data has to be RGB image'
        assert (len(image.shape) == 3), 'Input data has to be RGB image'
        assert np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'

        # First attempt to find qr codes - applies pyzbar decode function on raw image

        # Reading QR code in image
        qr_code = pz.decode(image)

        # Second attempt to find qr codes - applies pyzbar decode function on grayscale image transformation
        if not qr_code:

            # Transform data to grayscale color space
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Reading QR code in grayscale image
            qr_code = pz.decode(gray)

        # Next set of attempts work with cropped qr code from sector area (this is done with find_qr_code_location
        # function)
        if not qr_code:

            qr_location_output = self.find_qr_code_location(image, size=(330,110), stride= 5,  low_thresh=np.array([50, 0, 150]), up_thresh= np.array([250,150,250]))

            # If qr code was localized
            if qr_location_output:

                h, w = image.shape[0:2]

                left = max(0, qr_location_output[0] - 10)
                right = min(w, qr_location_output[2] + 10)
                up = max(0, qr_location_output[1] - 10)
                down = min(h, qr_location_output[3] + 10)

                # Third attemp - resize qr code area to bigger image

                # Crop qr code area
                qr_code_area = image[up:down, left:right, :]

                big_qr_code_area = cv2.resize(qr_code_area, (qr_code_area.shape[0] * 4, qr_code_area.shape[0] * 4))

                # Reading QR code
                qr_code = pz.decode(big_qr_code_area)

                # Fourth attempt - draw board around code area, crop qr code area with the board
                if not qr_code:

                    im = image.copy()

                    # Draw qr code area board
                    cv2.rectangle(im, (qr_location_output[0], qr_location_output[1]),
                                    (qr_location_output[2], qr_location_output[3]), (0, 255, 0), 3)

                    # Crop qr code area
                    qr_code_area_painted = im[up:down, left:right, :]

                    # Reading QR code
                    qr_code = pz.decode(qr_code_area_painted)

                # Fifth attempt - resize painted qr code area to bigger image
                elif not qr_code:

                    big_qr_code_area_painted = cv2.resize(qr_code_area_painted,
                                                            (qr_code_area_painted.shape[0] * 4,
                                                            qr_code_area_painted.shape[0] * 4))

                    # Reading QR code
                    qr_code = pz.decode(big_qr_code_area_painted)

                # Sixth attempt - sharpens qr code area with sharp_image function
                elif (not qr_code):

                    gray_qr_code_area = cv2.cvtColor(qr_code_area, cv2.COLOR_BGR2GRAY)
                    sharp_qr_code_area = self.sharp_image(gray_qr_code_area, 5, 0.1, 'kernel')

                    # Reading QR code
                    qr_code = pz.decode(sharp_qr_code_area)


                # Seventh attempt - resize painted qr code area to bigger image
                elif (not qr_code):

                    big_sharp_qr_code_area = cv2.resize(sharp_qr_code_area,
                                                        (qr_code_area_painted.shape[0] * 4,
                                                            qr_code_area_painted.shape[0] * 4))

                    # Reading QR code
                    qr_code = pz.decode(big_sharp_qr_code_area)


        if (qr_code):

            data = qr_code[0].data.decode('UTF-8')

        else:

            data = None

        return data

        #except Exception as e: print(e)


    def executor(self):

        #try:

        assert os.path.exists(self.filepath + 'images/'), 'There is not image folder'
        
        #Get list  of image names in data folder
        files = [file for file in os.listdir(self.filepath + 'images/')]

        flag = False

        #Some magical constants for roi determination according to something
        tray_count = len(files)/69
        system_break = tray_count*36

        #Get images locations and dates
        ids, dates, filenames = self.name2id()

        #Get unique trays in experiment according to camera position
        unique_trays = list(np.unique(ids))
        variants = []

        for k in range(0,len(unique_trays)): variants.append([unique_trays[k],0])

        print('*********************************************')
        print('CROPPING IMAGES AND REMOVING LENSDISTORT')
        print('*********************************************')

        for i, image in enumerate(files):

            img = cv2.imread(self.filepath + 'images/' + image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print('Cropping image ' + image)
            print('......' + str(i) + ' out of ' + str(len(files)))

            if self.chamber == 'xyz':

                img_dist = self.lensdistort(img, k=-0.1, model_number = 4, bordertype = 'crop')

                if i <= system_break: # tohle chyta tu zmenu uprostred experimentu

                    tray_frame = (10, 1840, 140, 2300)

                else:

                    tray_frame = (10+27, 1840+27, 140, 2300)

                #Manual roi cropping
                roi = img_dist[tray_frame[0]:tray_frame[1], tray_frame[2]:tray_frame[3],:]

                #Barcode decoding
                #Crop barcode
                barcode_roi = roi[800:1300, :150, :]
                tray_code = self.qr_code_reader(barcode_roi)

            elif self.chamber == 'hydropony':

                roi = img.copy()
                barcode_roi = roi[600:1300, 2700:, :]
                tray_code = self.qr_code_reader(barcode_roi)

            if tray_code and len(tray_code) == 9:

                print('Code of plate: ' + tray_code)
                variant = tray_code[6:9]
                variant_num = int(variant)
                newID = image[1:len(image)-9] + variant +  '.png'

                #file renaming according to barcode value
                year = tray_code[2:4]
                exp_num = tray_code[:1]
                typ = tray_code[4:6]
                exp_name = 'xy' + year + '-' + exp_num

                index = [x for x, y in enumerate(variants) if y[0] == ids[i]][0]

                if variant_num !=0 and flag==False:
                    
                    variants[index][1] = variant_num

                    if not any([item for item in variants if item[1] == 0]): flag = True

            else:

                print('Code of tray: not decoded')
                exp_name = None

            if exp_name and not os.path.exists(self.filepath + exp_name + '/'):

                os.mkdir(self.filepath + exp_name + '/')

            if not os.path.exists(self.filepath +  'smallimages/'):

                os.mkdir(self.filepath +  'smallimages/')
            
            file_path =  self.filepath +  'smallimages/' + image
            cv2.imwrite(file_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
    
        variants_df = pd.DataFrame(variants)
        variants_df.columns = ['location','variant']
        variants_df.to_excel(self.filepath + 'variants.xlsx')

        print('*********************************************')
        print('DONE: RESULTS WRITTEN TO SMALLIMAGES FOLDER')
        print('*********************************************')

        #except Exception as e: print(e)

############################################################################################################################################ 
if __name__ == "__main__":

    #Initialize image cropper
    IC = ImageCropper(gv.filepath, gv.chamber)
    IC.executor()