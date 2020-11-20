import numpy as np
from datetime import datetime
import cv2
import os
import scipy.io
import global_variables as gv

class Stacker:

    def __init__(self, filepath, pattern_path, chamber, row_num, col_num):

        self.filepath = filepath
        self.pattern_path = pattern_path
        self.chamber = chamber
        self.row_num = row_num
        self.col_num = col_num


    def name2id(self):

        try:

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

        except Exception as e: print(e)


    def create_cross_struct(self, image, MASK, cross_pattern):

        try:
                
            #image size
            h,w,d = image.shape

            #sempty structure for grid
            cross_pattern_struct = np.zeros((h,w))

            #shift of pattern
            ch,cw = cross_pattern.shape
            y_shift, x_shift = round(ch/2), round(cw/2)

            #mask boarder
            boarder = MASK[0][0][3][0]

            #grid coordinates
            grid_x_coords = MASK[0][0][2][:,0] + boarder[0]
            grid_y_coords = MASK[0][0][2][:,1] + boarder[2]

            for i in range(0,grid_x_coords.shape[0]):

                ys = grid_y_coords[i] - y_shift
                ye = grid_y_coords[i] + y_shift
                xs = grid_x_coords[i] - x_shift
                xe = grid_x_coords[i] + x_shift - 1

                cross_pattern_struct[ys:ye, xs:xe] = cross_pattern

            return cross_pattern_struct
    
        except Exception as e: print(e)


    def executor(self):

        class plant:

            def __init__(self, row, column, time, image, mask):

                self.row = row
                self.column = column
                self.time = time
                self.image = image
                self.mask = mask

        if not os.path.exists(self.filepath + 'stack/'): os.mkdir(self.filepath + 'stack/')

        margin = 0

        #Get list  of image names in data folder
        files = [file for file in os.listdir(self.filepath + 'smallimages/')]

        #Load computed trays masks
        MASKS = scipy.io.loadmat(self.filepath + 'masks.mat')['tray_attributes']

        #Load matrix pattern for grid identification
        cross_pattern = scipy.io.loadmat(self.pattern_path + 'etalon.mat')['krizek']

        #Get unique trays from data structure
        locations = []
        for i in range(0,MASKS.shape[1]): locations.append(MASKS[0,i][0][0][1][0])

        #Get images locations and dates
        ids, dates, file_names = self.name2id()

        #Get unique trays in experiment according to camera position
        unique_trays = list(np.unique(ids))

        for i, tray in enumerate(unique_trays):

            print('*********************************************')
            print('ASSEMBLING TRAY: ' + tray)

            tray_images = [files[index] for index in [i for i, item in enumerate(ids) if item == tray]]
            MASK = MASKS[0,[i for i, loc in enumerate(locations) if loc == tray][0]]

            tray_stack = []

            #Load first image
            image = cv2.imread(self.filepath + 'smallimages/' + tray_images[0])

            #Mask boarder
            boarder = MASK[0][0][3][0]
            
            #Create cross structure
            cross_struct = self.create_cross_struct(image, MASK, cross_pattern)

            for image in tray_images:
                print(image)
                image_stack = []

                time = [dates[index] for index in [i for i, item in enumerate(files) if item == image]][0]
                img = cv2.imread(self.filepath + 'smallimages/' + image)
                h,w = img.shape[0:2]
                #grid coordinates
                grid_x_coords = np.round(MASK[0][0][2][:,0] + boarder[0] -1)
                grid_y_coords = np.round(MASK[0][0][2][:,1] + boarder[2] -1)

                rows = np.fliplr(grid_y_coords.reshape(self.col_num+1,self.row_num+1))
                columns = np.transpose(grid_x_coords.reshape(self.col_num+1,self.row_num+1))
                
                #Separate image according to mask
                for r in range(0, rows.shape[1]-1):
                    for c in range(0, columns.shape[1]-1):
                        
                        #well location coordinates
                        sr = max(0,rows[c,r]-margin)
                        er = min(rows[c,r+1] + margin,h)
                        sc = max(0,columns[r,c] - margin)
                        ec = min(columns[r,c+1] + margin,w)

                        
                        #crop well from the image
                        well = img[sr:er,sc:ec,:]

                        #define mask of area for plant detection
                        mask = (cross_struct == 0).astype('uint8')[sr:er,sc:ec]

                        mh,mw = mask.shape
                        x_buf = margin + 10
                        y_buf = margin + 3 
                        mask[:y_buf,:] = 0
                        mask[mh-y_buf:,:] = 0
                        mask[:,:x_buf] = 0
                        mask[:,mw-x_buf:] = 0
                        
                        image_stack.append(plant(r,c,datetime.timestamp(time),well,mask))

                tray_stack.append(image_stack)

            scipy.io.savemat(self.filepath + 'stack/' +  '{}_stack_data.mat'.format(tray), mdict={'tray_stack': tray_stack})




############################################################################################################################################ 
if __name__ == "__main__":


    #Initialize Stacker
    stacker = Stacker(gv.filepath, gv.pattern_path, gv.chamber, gv.row_num, gv.col_num)
    stacker.executor()