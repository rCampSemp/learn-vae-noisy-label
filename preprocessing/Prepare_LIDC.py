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
    def __init__(self, LIDC_path, save_path, clevel, mask_threshold, resolution, meta_keys) -> None:
        self.LIDC_path = LIDC_path
        self.save_path = save_path

        self.patient_ids = []
        self._get_patients(LIDC_path)

        self.keylist = meta_keys
        self.metadata = defaultdict(list) 
        self.res = resolution
        self.mask_threshold = mask_threshold
        self.clevel = clevel

    def config(self):
        """Function to write LIDC data location to config file for pylidc.

        Args:
            path_to_lidc (str): absoluate file path to LIDC dataset parent folder 
        """
        # write path to pylidc config file
        f = open('/home/rhys/.pylidcrc', 'w')
        f.write(f'[dicom]\npath = {self.LIDC_path}\nwarn = True\n\n')
        f.close()

    def make_folders(self):
        os.makedirs(self.save_path, exist_ok=True)

        self._make_split_folder('train')
        self._make_split_folder('validate')
        self._make_split_folder('test')

        meta_path = self.save_path + '/meta'
        os.makedirs(meta_path, exist_ok=True)
    
    def _make_split_folder(self, split):
        annot_path = self.save_path + '/' + split + '/masks/annots'
        truth_path = self.save_path + '/' + split + '/masks/GT'
        scan_path = self.save_path + '/' + split + '/scans'

        os.makedirs(annot_path, exist_ok=True)
        os.makedirs(truth_path, exist_ok=True)
        os.makedirs(scan_path, exist_ok=True)

    def _get_patients(self, lidc_path):
        for fname in glob.glob(lidc_path + 'LIDC-IDRI-*'):
            self.patient_ids.append(fname[-14:])
        
        random.shuffle(self.patient_ids)

    def train_test_val_split(self):
        """Splits input file path into train/test/validation splits based on ratio.

        Args:
            allFiles (list[str]): list of strings representing file paths to data

        Returns:
            lists of ndarrays: lists of sub-arrays representing train/test/validation split of input file paths
        """
        testsplit = 0.15
        valsplit = 0.15

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
            dim (int): desired dimension to resize bbox to e.g. dim=64 corresponds to scan volume size 64x64

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
        """_summary_

        Args:
            annots (:obj: 'list' of :obj: 'pylidc.Annotation'): list of pylidc.Annotation objects that refer to the same physical nodule in the scan
            params: Variable length list of meta information keys

        Returns:
            list of floats: list of average scores for each annotation for a nodule correspond to their keys in *params
        """
        averages = []
        for param in self.keylist[2:]:
            tmp_score = 0
            for annot in annots:
                tmp_score += getattr(annot, param)
        
            averages.append(round(tmp_score/len(annots)))

        return averages

    def _standardize(self, img, slice_idx):
        std_img = (img[:,:,:,slice_idx] - img[:,:,:,slice_idx].mean() ) / img[:,:,:,slice_idx].std()
        return std_img

    def save_ims(self, data, path_to_split):
        """_summary_

        Args:
            split_data (_type_): _description_
            truepath (_type_): _description_
            annotpath (_type_): _description_
            scanpath (_type_): _description_
        """        
        annotpath = path_to_split + '/masks/annots'
        truepath = path_to_split + '/masks/GT'
        scanpath = path_to_split + '/scans'

        for pid in data:
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            vol = scan.to_volume() 
            nods = scan.cluster_annotations()
            #
            if len(nods) > 0:

                for nod_idx, nod in enumerate(nods):
                    # get consensus masks and bounding box 
                    cmask,cbbox,masks = consensus(nod, clevel=self.clevel, pad=None, ret_masks=True, verbose=False)
                    #
                    # ignore nodules too large
                    if cmask.shape[0] > self.res or cmask.shape[1] > self.res:
                        continue
                    #
                    # reshape bounding box to given dimensions
                    cbbox2 = self.reshape(cbbox)
                    scan_img = vol[cbbox2]
                    #
                    numslices = scan_img.shape[2]
                    # some scans have no slices
                    if numslices <= 0:
                        continue
                    #
                    # add channel dim to data  
                    scan_img = np.expand_dims(scan_img, axis=0)
                    cmask = np.expand_dims(cmask, axis=0)
                    for annot in masks:
                        annot.resize(1, cmask.shape[1], cmask.shape[2], numslices, refcheck=False) #refcheck false to allow resizing referenced array
                    #
                    #
                    # stack each annotation from list of annotations to c x h x w x annots x slices
                    masks = np.stack(masks, axis=3)
                    #
                    # pad masks into given dimensions
                    cmask = np.pad(cmask, pad_width=((0,0), (cbbox[0].start - cbbox2[0].start, cbbox2[0].stop - cbbox[0].stop), (cbbox[1].start - cbbox2[1].start, cbbox2[1].stop - cbbox[1].stop), (0,0)), mode='constant', constant_values=False)
                    masks = np.pad(masks, pad_width=((0,0), (cbbox[0].start - cbbox2[0].start, cbbox2[0].stop - cbbox[0].stop), (cbbox[1].start - cbbox2[1].start, cbbox2[1].stop - cbbox[1].stop), (0,0), (0,0)), mode='constant', constant_values=False)
                    #
                    # save metadata to dict
                    averages = self.getAverageParams(nod)
                    data = [pid[-4:], nod_idx] + averages
                    for i, key in enumerate(self.keylist):
                        self.metadata[key].append(data[i])
                    #
                    for slice_idx in range(numslices):
                        if np.sum(cmask[:,:,:,slice_idx]) <= self.mask_threshold or cmask.shape[1] != self.res or cmask.shape[2] != self.res:
                            continue
                        # save ground truth masks
                        full_store_path_true = os.path.join(truepath, 'pid_' + pid[-4:] + '_nod_' + str(nod_idx) 
                                                                + '_slice_' + str(slice_idx) + '.tif')
                        imsave(full_store_path_true, cmask[:,:,:,slice_idx])
                    
                        #save scan of lung
                        std_scan_img = self._standardize(scan_img, slice_idx)
                        full_store_path_scan = os.path.join(scanpath, 'pid_' + pid[-4:] + '_nod_' + str(nod_idx) 
                                                                + '_slice_' + str(slice_idx) + '.tif')
                        imsave(full_store_path_scan, std_scan_img)

                        # save masks per slice with multiple annots along 3rd dim h x w x annots
                        full_store_path_annot = os.path.join(annotpath, 'pid_' + pid[-4:] + '_nod_' + str(nod_idx) 
                                                                + '_slice_' + str(slice_idx) + '.tif')
                        imsave(full_store_path_annot, masks[:,:,:,:,slice_idx])
    
    def save_meta_csv(self):
        meta_df = pd.DataFrame.from_dict(self.metadata)
        meta_df.sort_values(by=['patient_id', 'nodule_no'], inplace=True)
        df_filename = 'metadata.csv'
        metadf_path = os.path.join('../LIDC_examples/meta', df_filename)
        meta_df.to_csv(metadf_path, index=False)

    def prepare_data(self):
        self.config()
        self.make_folders()

        train_files, val_files, test_files = self.train_test_val_split()

        self.save_ims(train_files, self.save_path + '/train')
        self.save_ims(test_files, self.save_path + '/test')
        self.save_ims(val_files, self.save_path + '/validate')

        self.save_meta_csv()

    
if __name__ == '__main__':
    keylist = ['patient_id','nodule_no','subtlety', 'internalStructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']

    path_lidc = '/home/rhys/Documents/datasets/LIDC-IDRI/'
    save_path = '../LIDC_examples'

    lidc_process = PrepareLIDC(path_lidc, save_path, clevel=0.5, mask_threshold=30, resolution=64, meta_keys=keylist)
    lidc_process.prepare_data()
