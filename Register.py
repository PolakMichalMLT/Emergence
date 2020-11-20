import numpy as np
import pandas as pd
from datetime import datetime
import cv2
import os
import scipy.io
from skimage.feature import peak_local_max
from sklearn.linear_model import LinearRegression
from scipy.ndimage import median_filter
import global_variables as gv
import warnings

warnings.filterwarnings("ignore")

class TrayRegister:

    def __init__(self, filepath, pattern_path, chamber, row_num, col_num):

        self.filepath = filepath
        self.pattern_path = pattern_path
        self.chamber = chamber 
        self.row_num = row_num
        self.col_num = col_num


    def name2id(self):

        #try:

        #Get list  of image names in data folder
        file_list = [file for file in os.listdir(self.filepath + 'smallimages/')]

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

    def segmentation(self, image, low_thresh, up_thresh):

        try:
            
            # Verify correctness of input arguments
            assert (type(image) == np.ndarray), 'Input data has to be RGB image'
            assert (len(image.shape) == 3), 'Input data has to be RGB image'
            assert np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'

            assert (type(low_thresh) == np.ndarray) & (len(low_thresh)) & (np.min(low_thresh) >= 0) & (
                        np.max(low_thresh) <= 255), 'Low_thresh input argument is not valid'
            assert (type(up_thresh) == np.ndarray) & (len(up_thresh)) & (np.min(up_thresh) >= 0) & (
                        np.max(up_thresh) <= 255), 'Up_thresh input argument is not valid'

            # Transform RGB image to HSV color space
            imageHSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # mask for mgreen segmentation
            maskHSV = cv2.inRange(imageHSV, low_thresh, up_thresh)

            # Determine, which pixels are biomass of microgreen
            logic = maskHSV > 0
            # Transform boolean values to integers
            mask = logic.astype('uint8')

            return mask

        except Exception as e: print(e)


    def crop_tray(self, image):

        try:

            if self.chamber == 'xyz':

                r,g,b = cv2.split(image)
                mask = (((r+20)<b)*((g+20)<b)*(b<100)).astype('uint8')*255

                #columns projection
                column_projection = mask.sum(axis=0)
                #limit for column boarders
                column_limit = max(column_projection)/2

                #rows projection
                row_projection = mask.sum(axis=1)
                #limit for rows boarders
                row_limit = max(row_projection)/2

            elif self.chamber == 'hydropony':
                
                #Tady ten zpusob identifikace sadbovace neni spolehliva

                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                h,s,v = cv2.split(hsv)
                mask = (((h+10)<v)*(s<60))
                mask = (mask==0).astype('uint8')*255

                #columns projection
                column_projection = mask.sum(axis=0)
                #limit for column boarders
                column_limit = np.quantile(column_projection, 0.8)

                #rows projection
                row_projection = mask.sum(axis=1)
                #limit for rows boarders
                row_limit = np.quantile(row_projection, 0.85)


            h,w = mask.shape

            #Looking for edge on right side of tray
            edge_position = round(len(column_projection)/2)
            while column_projection[edge_position] < column_limit: edge_position += 1 
            right_edge = edge_position

            #Looking for edge on left side of tray
            edge_position = round(len(column_projection)/2)
            while column_projection[edge_position] < column_limit: edge_position -= 1 
            left_edge = edge_position

            #Looking for edge on left side of tray
            edge_position = round(len(row_projection)/2)
            while row_projection[edge_position] < row_limit: edge_position += 1 
            down_edge = edge_position

            #Looking for edge on left side of tray
            edge_position = round(len(row_projection)/2)
            while row_projection[edge_position] < row_limit: edge_position -= 1 
            top_edge = edge_position

            return (max(0,left_edge),min(right_edge,w), max(0,top_edge), min(down_edge,h))

        except Exception as e: print(e)


    def get_perspective_corrected_roi(self, image):

        try:

            h,w = image.shape[0:2]
            segmented = self.segmentation(image, np.array([160,40,120]), np.array([250,250,250]))
            segmented[:, :170] = 0
            segmented[:,w-140:] = 0
            mask = median_filter(segmented, 5)
            coords = np.where(mask > 0)
            points = []

            for i, coord in enumerate(coords[0]): points.append((coords[1][i], coords[0][i]))

            top_left = points[np.argmin(np.power(np.sum(np.power(np.array(points)-np.concatenate((np.repeat(0, len(points)).reshape(-1,1), np.repeat(0, len(points)).reshape(-1,1)), axis = 1),2), axis=1), 0.5))]
            top_right = points[np.argmin(np.power(np.sum(np.power(np.array(points)-np.concatenate((np.repeat(w, len(points)).reshape(-1,1), np.repeat(0, len(points)).reshape(-1,1)), axis = 1),2), axis=1), 0.5))]
            bottom_left = points[np.argmin(np.power(np.sum(np.power(np.array(points)-np.concatenate((np.repeat(0, len(points)).reshape(-1,1), np.repeat(h, len(points)).reshape(-1,1)), axis = 1),2), axis=1), 0.5))]
            bottom_right = points[np.argmin(np.power(np.sum(np.power(np.array(points)-np.concatenate((np.repeat(w, len(points)).reshape(-1,1), np.repeat(h, len(points)).reshape(-1,1)), axis = 1),2), axis=1), 0.5))]


            # We will first manually select the source points 
            # we will select the destination point which will map the source points in
            # original image to destination points in unwarped image
            src = np.float32([top_left, top_right, bottom_left, bottom_right])

            dst = np.float32([(0, 0), (w, 0) , (0, h), (w, h)])

            # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
            M = cv2.getPerspectiveTransform(src, dst)
            # use cv2.warpPerspective() to warp your image to a top-down view
            warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)

            rect = (min(top_left[0],bottom_left[0]), max(top_right[0],bottom_right[0]), min(top_left[1], top_right[1]) , max(bottom_left[1], bottom_right[1]))

            roi = cv2.resize(warped,(rect[1]-rect[0], rect[3]-rect[2]))

            return roi, rect

        except Exception as e: print(e)


    def find_grid(self, image):

        try:

            roi_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h,w = roi_gray.shape

            axis_x = np.floor(np.linspace(20, (w-20), self.col_num+1))
            axis_y = np.floor(np.linspace(50, (h-50), self.row_num+1))

            # Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(roi_gray,(5,5),0)
            threshold,binary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            #Filter of thresholded image with cross pattern
            cross_pattern = scipy.io.loadmat(self.pattern_path+'etalon.mat')['krizek']
            filter_intensity = cv2.filter2D(binary.astype(np.float32), -1, cross_pattern)

            #Find local peaks
            convolution = np.ones((3,3))
            peaks = (peak_local_max(filter_intensity, footprint=convolution, indices=False, exclude_border=0)) 

            #remove the boarder with noise
            h,w = peaks.shape
            x_buf = int(axis_x[1]//2)
            y_buf = int(axis_y[1]//2)

            peaks = cv2.copyMakeBorder(
                peaks.astype('uint8'),
                top=y_buf,
                bottom=y_buf,
                left=x_buf,
                right=x_buf,
                borderType=cv2.BORDER_CONSTANT,
                value=[0]
            )

            #Allocation of crosses centers into matrix
            coord_x = np.empty((axis_x.shape[0], axis_y.shape[0]))
            coord_y = np.empty((axis_x.shape[0], axis_y.shape[0]))
            coord_x[:] = np.NaN
            coord_y[:] = np.NaN

            coordinates = np.nonzero(peaks)

            for i in range(0,coord_x.shape[0]-1):
                for j in range(0,coord_y.shape[1]-1):

                    etal = (np.repeat(axis_y[j],coordinates[0].shape[0]),np.repeat(axis_x[i],coordinates[0].shape[0]))
                    distance = abs(etal[0] - coordinates[0]) + abs(etal[1] - coordinates[1])
                    position = np.argmin(distance)

                    if distance[position] < 50:

                        coord_x[i,j] = coordinates[1][position]
                        coord_y[i,j] = coordinates[0][position]

            return (coord_x, coord_y)

        except Exception as e: print(e)

    def loss(self, x, points, classes, row_num, col_num):

        try:
        
            angle = x[0]
            
            MAT = np.matrix([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            
            points = points*MAT
            cost = 0

            for i in range(0,row_num-1):
                                                                                        
                if np.sum(classes[:,0] == i):

                    cost =+ np.var(points[(classes[:,0] == i),0])

            for j in range(0,col_num-1):

                if np.sum(classes[:,1] == j):

                    cost =+ np.var(points[(classes[:,1] == j),1])

            return cost
        
        except Exception as e: print(e)


    def grid_optimizer(self, roi, grid):

        try:

            h,w = roi.shape[0:2]
            row_num, col_num = grid[0].shape

            axis_x = np.floor(np.linspace(20, (w-20), row_num))
            axis_y = np.floor(np.linspace(50, (h-50), col_num))

            #Create matrix of points
            points = np.concatenate((grid[0].flatten().reshape(-1,1),grid[1].flatten().reshape(-1,1)), axis = 1)

            #remove nan values
            points = points[~np.isnan(points[:,0])]

            #reverse y-point coordinate, so  pixel value is growing from bottom of image
            points[:,1] = h - points[:,1]

            # points[:,0] is x-axis from left to right
            # points[:,1] is y-axis from bottom to top
            classes = np.empty((points.shape))
            classes[:] = np.NaN

            for i in range(0,points.shape[0]):

                point = points[i,:]

                classes[i,0] = np.argmin(abs(point[0]-axis_x))
                classes[i,1] = np.argmin(abs(point[1]-axis_y))

            #find optimal grid rotation based on sum of inner-class variabilities
            opt_result = scipy.optimize.minimize(self.loss, x0 = [0], args=(points, classes, row_num, col_num))
            optimal_rotation_angle = opt_result.x[0]

            optimal_rotation_matrix = np.matrix([[np.cos(optimal_rotation_angle),-np.sin(optimal_rotation_angle)],
                                                [np.sin(optimal_rotation_angle),np.cos(optimal_rotation_angle)]])

            #Rotate points into new coordinates
            rotated_points = points*optimal_rotation_matrix

            #Frames for grid coordinates
            x_coords = np.empty(row_num)
            y_coords = np.empty(col_num)
            x_coords[:] = np.NaN
            y_coords[:] = np.NaN

            #For each row compute coordinate as a mean of available points
            for i in range(0,x_coords.shape[0]-1): x_coords[i] = np.mean(rotated_points[(classes[:,0] == i),0])

            #Smooth the line with linear regression and predict missing
            X = np.concatenate((np.ones(row_num).reshape(-1,1),np.arange(row_num).reshape(-1,1)), axis = 1)
            model = LinearRegression().fit(X[~np.isnan(x_coords),:], x_coords[~np.isnan(x_coords)])
            x_coords_new = model.predict(X)

            #For each row compute coordinate as a mean of available points
            for j in range(0,y_coords.shape[0]-1): y_coords[j] = np.mean(rotated_points[(classes[:,1] == j),1])

            #Smooth the line with linear regression and predict missing
            Y = np.concatenate((np.ones(col_num).reshape(-1,1),np.arange(col_num).reshape(-1,1)), axis = 1)
            model = LinearRegression().fit(Y[~np.isnan(y_coords),:], y_coords[~np.isnan(y_coords)])
            y_coords_new = model.predict(Y)

            #Compose new points
            new_points = np.empty((col_num*row_num,2))
            new_points[:] = np.NaN

            position = -1

            for i in range(0, row_num):
                for j in range(0, col_num):

                    position += 1
                    new_points[position,:] = np.array([x_coords_new[i],y_coords_new[j]])

            #rotate points back into original coordinates
            new_points_rotated = new_points * np.linalg.inv(optimal_rotation_matrix)

            #re-orient y-coordinate with origin at the top of image
            new_points_rotated[:,1] = h - new_points_rotated[:,1]

            new_points_rotated[np.squeeze(np.array((new_points_rotated[:,0] > w)).reshape(new_points_rotated.shape[0])), 0] = w
            new_points_rotated[np.squeeze(np.array((new_points_rotated[:,1] > h)).reshape(new_points_rotated.shape[0])), 1] = h

            return np.array(np.round(new_points_rotated), dtype=np.intc)

        except Exception as e: print(e)


    def executor(self):

        class mask:

            def __init__(self, filename, location, mask, boarder):

                self.filename = filename
                self.location = location
                self.mask = mask
                self.boarder = boarder

        #try:

        if not os.path.exists(self.filepath + 'masks/'): os.mkdir(self.filepath + 'masks/')

        #Get list  of image names in data folder
        files = [file for file in os.listdir(self.filepath + 'smallimages/')]

        #Get images locations and dates
        ids, dates, file_names = self.name2id())
        #Merge file names with dates
        merged_list = tuple(zip(file_names, dates, ids))
        sorted_merged_list = sorted(merged_list, key=lambda tup: tup[1])

        #Get unique trays in experiment according to camera position
        unique_trays = list(np.unique(ids))

        #Find first unique occurence of trays in experiment
        files_for_masks = []
        for ut in unique_trays: files_for_masks.append([(y[0],y[2]) for x, y in enumerate(sorted_merged_list) if y[2] == ut][0])

        #Limit of pixels for grid lines distance
        limit = 20

        masks = []


        for i,image in enumerate(files_for_masks):
            
            img = cv2.imread(self.filepath + 'smallimages/' + image[0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            print('Analyzing image ' + image[0] + '...')
            print('...' + str(i+1) + ' out of ' + str(len(files_for_masks)))


            #Cropping of black part of a tray
            if self.chamber == 'xyz':

                boarder = self.crop_tray(img)
                roi = img[boarder[2]:boarder[3],boarder[0]:boarder[1],:]

            elif self.chamber == 'hydropony':

                roi, boarder = self.get_perspective_corrected_roi(img)
                


            #Computation of initial location of cross pattern
            grid = self.find_grid(roi)
            
            #optimizing grid structure
            points = self.grid_optimizer(roi, grid)

            #Visualize grid and save
            paint = roi.copy()

            for point in points:

                paint = cv2.circle(paint, (int(point[0]), int(point[1])), 5, [255,0,0], thickness = 4)

            masks.append(mask(image[0], image[1], points, boarder))
            paint = cv2.cvtColor(paint, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.filepath + 'masks/' + image[0], paint)

        scipy.io.savemat(self.filepath + 'masks.mat', mdict={'tray_attributes': masks})

        #except Exception as e: print(e)



############################################################################################################################################ 
if __name__ == "__main__":

    #Initialize TrayRegister

    TR = TrayRegister(gv.filepath, gv.pattern_path, gv.chamber, gv.row_num, gv.col_num)
    TR.executor()