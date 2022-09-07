import os
import pylidc as pl
import numpy as np
from pylidc.utils import consensus
import glob
from tifffile import imsave
import random
import pandas as pd
from collections import defaultdict

class PrepareLIDC:
    """Class for preprocessing and saving our LIDC dataset with its' metadata.

    Args:
        LIDC_path (str): path to full unprocessed LIDC data with subfolders of DICOM files
        save_path (str): file path to save all data to
        clevel (float): agreement level of consensus consolidation
        mask_threshold (2-tuple of int): 1- lower limit of mask sizes we allow  2- upper limit of mask sizes we allow
        resolution (int): size of the resulting images we want, will result in saving images of size resolution x resolution
        meta_keys (list): list of str that are headers of parameters of the metadata associated with lidc data
        hist (str, optional): indicates type of regime to apply to CT scans
        cliplimit (float, optional): clip limit for CLAHE histogram equalization. Defaults to 0.2
    """
    def __init__(self, LIDC_path, save_path, clevel, mask_threshold, resolution, meta_keys, hist=None, cliplimit=0.2) -> None:
        """Constructor for PrepareLIDC class.
        """
        self.LIDC_path = LIDC_path
        self.save_path = save_path

        self.patient_ids = []
        self._get_patients(LIDC_path)

        self.keylist = meta_keys
        self.metadata = defaultdict(list) 
        self.res = resolution
        self.mask_threshold = mask_threshold
        self.clevel = clevel

        self.hist = hist
        self.cliplimit = cliplimit

    def config(self):
        """Function to write LIDC data location to config file for pylidc."""
        # write path to pylidc config file
        f = open('/home/rhys/.pylidcrc', 'w')
        f.write(f'[dicom]\npath = {self.LIDC_path}\nwarn = True\n\n')
        f.close()

    def make_folders(self):
        """Creates train/test/validate folders and their subfolders.
        """
        os.makedirs(self.save_path, exist_ok=True)

        self._make_split_folder('train')
        self._make_split_folder('validate')
        self._make_split_folder('test')

        meta_path = self.save_path + '/meta'
        os.makedirs(meta_path, exist_ok=True)
    
    def _make_split_folder(self, split):
        """makes subfolder for make_folders method

        Args:
            split (str): str to indicate type of subfolder to create
        """
        annot_path = self.save_path + '/' + split + '/masks/annots'
        truth_path = self.save_path + '/' + split + '/masks/GT'
        scan_path = self.save_path + '/' + split + '/scans'

        os.makedirs(annot_path, exist_ok=True)
        os.makedirs(truth_path, exist_ok=True)
        os.makedirs(scan_path, exist_ok=True)

    def _get_patients(self, lidc_path):
        """Collects all patient ids available and shuffles in random order. 

        Args:
            lidc_path (str): path to unprocessed LIDC data 
        """
        for i, fname in enumerate(glob.glob(lidc_path + 'LIDC-IDRI-*')):
            self.patient_ids.append(fname[-14:])
        
        random.shuffle(self.patient_ids)

    def train_test_val_split(self):
        """Splits input file path into train/test/validation splits based on ratio.

        Returns:
            lists of ndarrays: lists of sub-arrays representing train/test/validation split of input file paths
        """
        testsplit = 0.1
        valsplit = 0.1

        train_Files, val_Files, test_Files = np.split(np.array(self.patient_ids),
                                                        [int(len(self.patient_ids) * (1 - (valsplit + testsplit))),
                                                        int(len(self.patient_ids) * (1 - valsplit)),
                                                        ])

        return train_Files, val_Files, test_Files
    
    def reshape(self, bbox):
        """
        Reshapes input bounding box to crop LIDC scans to desired size.

        Args:
            bbox (tuple of slice objects): 3-tuple of Python slice objects around nodule of interest 

        Returns:
            tuple: 3-tuple of Python slice objects to index the image volume in desired dim size
        """
        x1, x2, y1, y2 = bbox[0].start, bbox[0].stop, bbox[1].start, bbox[1].stop
        
        # find centre of image
        xcentre = (x1 + x2) // 2
        ycentre = (y1 + y2) // 2

        # find min and max x values for bounding box
        bboxx1 = xcentre - (self.res // 2) 
        bboxx2 = xcentre + (self.res // 2) 

        # find min and max y values for bounding box
        bboxy1 = ycentre - (self.res // 2) 
        bboxy2 = ycentre + (self.res // 2) 

        return slice(bboxx1, bboxx2), slice(bboxy1, bboxy2), bbox[2]

    def getAverageParams(self, annots):
        """Calculates list of average parameters from metadata on LIDC data from keylist

        Args:
            annots (list of pylidc.Annotation's): list of pylidc.Annotation objects that refer to the same physical nodule in the scan
            params: Variable length list of meta information keys

        Returns:
            list of floats: list of average scores for each annotation for a nodule correspond to their keys in *params
        """
        # iterate over keylist
        averages = []
        for param in self.keylist[2:]:
            tmp_score = 0
            for annot in annots:
                tmp_score += getattr(annot, param) # get value of parameter from keylist from anntotation

            # calculate average and add to list
            averages.append(round(tmp_score/len(annots)))

        return averages

    def _standardize(self, img):
        """Z-score standardisation of img input.

        Args:
            img (numpy array): unnormalised scan image of lidc 

        Returns:
            std_img (numpy array): standardised img array
        """
        std_img = ( img - img.mean(axis=(0,1), keepdims=True) ) / img.std(axis=(0,1), keepdims=True)
        return std_img

    # def zeroto255(self, img):
    #     new_img = ( ( img - img.min() ) * ( 1/(img.max() - img.min() ) * 255) ).astype(np.uint8)
    #     return new_img

    # def clahe_fn(self, img):
    #     new_img = ( ( img - img.min() ) * ( 1/(img.max() - img.min() ) * 255) ).astype(np.uint8)
    #     clahe = cv.createCLAHE(clipLimit=self.cliplimit, tileGridSize=(8,8))
    #     new_img = clahe.apply(new_img)
    #     return new_img

    # def equal_hist(self, img):
    #     new_img = ( ( img - img.min() ) * ( 1/(img.max() - img.min() ) * 255) ).astype(np.uint8)
    #     eq_img = cv.equalizeHist(new_img)
    #     return eq_img

    def save_ims(self, data, path_to_split):
        """Main loop to save images and binary masks.

        Args:
            data (list): list of str denoting patient IDs to save
            path_to_split (str): folder path to scans to save
        """       
        # create subfolder file paths  
        annotpath = path_to_split + '/masks/annots'
        truepath = path_to_split + '/masks/GT'
        scanpath = path_to_split + '/scans'

        # iterate over each patient in data list
        for pid in data:
            
            # query for CT scans with patient id = pid and convert to an array volume
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            vol = scan.to_volume() 

            # cluster the annotations of the scan
            nods = scan.cluster_annotations()
            
            # if nodules are present
            if len(nods) > 0:

                # iterate over all nodules in cluster
                for nod_idx, nod in enumerate(nods):

                    # get consensus, bounding box and annotations
                    cmask,cbbox,masks = consensus(nod, clevel=self.clevel)
                    
                    # require more than one annotation per nodule
                    if len(masks) < 2:
                        continue
                    
                    # ignore nodules too large and unable to rescale to size
                    if cmask.shape[0] > self.res or cmask.shape[1] > self.res:
                        continue
                    
                    # reshape bounding box to given dimensions
                    cbbox2 = self.reshape(cbbox)
                    scan_img = vol[cbbox2]

                    # some scans have no slices need to be ignored
                    numslices = scan_img.shape[2]
                    if numslices <= 0:
                        continue
                    
                    # add channel dimension to data  
                    cmask = np.expand_dims(cmask, axis=0)
                    for annot in masks:
                        annot.resize(1, cmask.shape[1], cmask.shape[2], numslices, refcheck=False) #refcheck false to allow resizing referenced array
                    
                    # stack each annotation from list of annotations to c x h x w x annots x slices
                    masks = np.stack(masks, axis=3)
                    
                    # pad masks into given dimensions
                    cmask = np.pad(cmask, pad_width=((0,0), (cbbox[0].start - cbbox2[0].start, cbbox2[0].stop - cbbox[0].stop), (cbbox[1].start - cbbox2[1].start, cbbox2[1].stop - cbbox[1].stop), (0,0)), mode='constant', constant_values=False)
                    masks = np.pad(masks, pad_width=((0,0), (cbbox[0].start - cbbox2[0].start, cbbox2[0].stop - cbbox[0].stop), (cbbox[1].start - cbbox2[1].start, cbbox2[1].stop - cbbox[1].stop), (0,0), (0,0)), mode='constant', constant_values=False)
                    
                    # save metadata to dict
                    averages = self.getAverageParams(nod)
                    data = [pid[-4:], nod_idx] + averages
                    for i, key in enumerate(self.keylist):
                        self.metadata[key].append(data[i])
                    
                    # empty list to hold nodule sizes for future analysis
                    nod_size_list = []

                    #   iterate over slices
                    for slice_idx in range(numslices):
                        nod_size = np.sum(cmask[:,:,:,slice_idx])

                        # skip slice if nodule size is too large or small or if scan image or GT does not fit resolution
                        if nod_size <= self.mask_threshold or nod_size >= 800 or cmask.shape[1] != self.res or cmask.shape[2] != self.res or scan_img.shape[0] != self.res or scan_img.shape[1] != self.res:
                            continue

                        # append nodule size to list
                        nod_size_list.append(nod_size)

                        # save ground truth masks
                        full_store_path_true = os.path.join(truepath, 'pid_' + pid[-4:] + '_nod_' + str(nod_idx) 
                                                                + '_slice_' + str(slice_idx) + '.tif')
                        imsave(full_store_path_true, cmask[:,:,:,slice_idx])
                    
                        ## save scan of lung
                        # method to process scan img
                        if self.hist is None:
                            new_scan_img = self._standardize(scan_img[:,:,slice_idx])
                        elif self.hist == 'clahe':
                            new_scan_img = self.clahe_fn(scan_img[:,:,slice_idx])
                        elif self.hist == 'equal':                            
                            new_scan_img = self.equal_hist(scan_img[:,:,slice_idx])
                        
                        # add channel dimension to img
                        new_scan_img = np.expand_dims(new_scan_img, axis=0)
                        full_store_path_scan = os.path.join(scanpath, 'pid_' + pid[-4:] + '_nod_' + str(nod_idx) 
                                                                + '_slice_' + str(slice_idx) + '.tif')
                        imsave(full_store_path_scan, new_scan_img)

                        # save masks per slice with multiple annots along 3rd dim h x w x annots
                        full_store_path_annot = os.path.join(annotpath, 'pid_' + pid[-4:] + '_nod_' + str(nod_idx) 
                                                                + '_slice_' + str(slice_idx) + '.tif')
                        imsave(full_store_path_annot, masks[:,:,:,:,slice_idx])

                    # add nodule sizes to metadata
                    self.metadata['nod_sz_per_slice'].append(nod_size_list)
    
    def save_meta_csv(self):
        """Saves metadata as scv
        """
        meta_df = pd.DataFrame.from_dict(self.metadata)
        meta_df.sort_values(by=['patient_id', 'nodule_no'], inplace=True)
        df_filename = 'metadata.csv'
        metadf_path = os.path.join(self.save_path + '/meta', df_filename)
        meta_df.to_csv(metadf_path, index=False)
        print('metadata saved')

    def prepare_data(self):
        """Main function to save all data and metadata
        """
        self.config()
        self.make_folders()

        train_files, val_files, test_files = self.train_test_val_split()

        self.save_ims(train_files, self.save_path + '/train')
        self.save_ims(test_files, self.save_path + '/test')
        self.save_ims(val_files, self.save_path + '/validate')

        self.save_meta_csv()

    
if __name__ == '__main__':
    # keylist parameters taken from pylidc documentation
    keylist = ['patient_id','nodule_no','subtlety', 'internalStructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']

    # path to unprocessed lidc data
    path_lidc = '/home/rhys/Documents/datasets/full_LIDC/LIDC-IDRI/'
    # save path where saved data will end up
    save_path = '../LIDC_examples'

    # save eeverything
    lidc_process = PrepareLIDC(path_lidc, save_path, clevel=0.5, mask_threshold=(30, 800), resolution=64, meta_keys=keylist, hist=None)
    lidc_process.prepare_data()

    print('End')

