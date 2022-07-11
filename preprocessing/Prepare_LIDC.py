from distutils.command.config import config
import os
import errno
from tabnanny import verbose
import pylidc as pl
import numpy as np
from pylidc.utils import consensus
import glob
from tifffile import imsave
import random


def prep_config(path_to_lidc):
    # path to root folder of LIDC dataset with subfolders in form LIDC-IDRI-dddd
    # path = '/home/rhys/Documents/datasets/LIDC-IDRI/'

    # write path to pylidc config file
    # f = open('/root/.pylidcrc', 'w')
    f = open('/home/rhys/.pylidcrc', 'w')
    f.write(f'[dicom]\npath = {path_to_lidc}\nwarn = True\n\n')
    f.close()

def train_test_val_split(allFiles):
    testsplit = 0.1
    valsplit = 0.1

    train_Files, val_Files, test_Files = np.split(np.array(allFiles),
                                                    [int(len(allFiles) * (1 - (valsplit + testsplit))),
                                                    int(len(allFiles) * (1 - valsplit)),
                                                    ])

    return train_Files, val_Files, test_Files

def get_patients(path_to_lidc):
    patient_ids = [fname[-14:] for fname in glob.glob(path_to_lidc + 'LIDC-IDRI-*')]
    return patient_ids

def reshape(bbox, dim):
    """ for even dim only """
    x1, x2, y1, y2 = bbox[0].start, bbox[0].stop, bbox[1].start, bbox[1].stop
    
    # find centre of image
    xcentre = (x1 + x2) // 2
    ycentre = (y1 + y2) // 2

    # find min and max x values for bounding box
    bboxx1 = xcentre - (dim[0] // 2) 
    bboxx2 = xcentre + (dim[0] // 2) 

    # find min and max y values for bounding box
    bboxy1 = ycentre - (dim[1] // 2) 
    bboxy2 = ycentre + (dim[1] // 2) 

    return slice(bboxx1, bboxx2), slice(bboxy1, bboxy2), bbox[2]

def save_ims(files, truepath, annotpath, scanpath):
    mask_threshold = 20
    for pid in files:
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
        vol = scan.to_volume() 
        nods = scan.cluster_annotations()
        no_nods = len(nods)
        #
        if no_nods > 0:

            for nod_idx, nod in enumerate(nods):
                # get consensus masks and bounding box 
                cmask,cbbox,masks = consensus(nod, clevel=0.5)
                #
                # ignore incorrectly reshaped imgs
                if cmask.shape[0] > 64 or cmask.shape[1] > 64:
                    continue
                #
                # reshape bounding box to given dimensions
                cbbox2 = reshape(cbbox, (64,64))
                scan_img = vol[cbbox2]
                #
                numslices = scan_img.shape[2]
                #
                # add channel dim to data  
                scan_img = np.expand_dims(scan_img, axis=0)
                cmask = np.expand_dims(cmask, axis=0)
                for annot in masks:
                    annot.resize(1, cmask.shape[1], cmask.shape[2], numslices, refcheck=False) #refcheck false to allow resizing referenced array
                #
                # some scans have no slices
                if numslices <= 0:
                    continue
                #
                # stack each annotation from list of annotations to c x h x w x annots x slices
                masks = np.stack(masks, axis=3)
                #
                # pad masks into given dimensions
                cmask = np.pad(cmask, pad_width=((0,0), (cbbox[0].start - cbbox2[0].start, cbbox2[0].stop - cbbox[0].stop), (cbbox[1].start - cbbox2[1].start, cbbox2[1].stop - cbbox[1].stop), (0,0)), mode='constant', constant_values=False)
                masks = np.pad(masks, pad_width=((0,0), (cbbox[0].start - cbbox2[0].start, cbbox2[0].stop - cbbox[0].stop), (cbbox[1].start - cbbox2[1].start, cbbox2[1].stop - cbbox[1].stop), (0,0), (0,0)), mode='constant', constant_values=False)
                # check for negative bbox
                #
                for slice_idx in range(numslices):
                    if slice_idx > 40 or np.sum(cmask[:,:,:,slice_idx]) <= mask_threshold or cmask.shape[1] != 64 or cmask.shape[2] != 64:
                        continue
                    # save ground truth masks
                    full_store_path_true = os.path.join(truepath, 'pid_' + pid[-4:] + '_nod_' + str(nod_idx) 
                                                            + '_slice_' + str(slice_idx) + '.tif')
                    imsave(full_store_path_true, cmask[:,:,:,slice_idx])
                   
                    #save scan of lung
                    full_store_path_scan = os.path.join(scanpath, 'pid_' + pid[-4:] + '_nod_' + str(nod_idx) 
                                                            + '_slice_' + str(slice_idx) + '.tif')
                    imsave(full_store_path_scan, scan_img[:,:,:,slice_idx])

                    # save masks per slice with multiple annots along 3rd dim h x w x annots
                    full_store_path_annot = os.path.join(annotpath, 'pid_' + pid[-4:] + '_nod_' + str(nod_idx) 
                                                            + '_slice_' + str(slice_idx) + '.tif')
                    imsave(full_store_path_annot, masks[:,:,:,:,slice_idx])
                


# save folder = '../LIDC_examples
def prep_data(path_to_lidc, save_folder_mother):
    try:
        os.makedirs(save_folder_mother)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

    image_train = save_folder_mother + '/train/masks/annots'
    truth_train = save_folder_mother + '/train/masks/GT'
    image_val = save_folder_mother + '/validate/masks/annots'
    truth_val = save_folder_mother + '/validate/masks/GT'
    image_test = save_folder_mother + '/test/masks/annots'
    truth_test = save_folder_mother + '/test/masks/GT'

    scan_train_path = save_folder_mother + '/train/scans'
    scan_val_path = save_folder_mother + '/validate/scans'
    scan_test_path = save_folder_mother + '/test/scans'

    try:
        os.makedirs(image_train)
        os.makedirs(truth_train)
        os.makedirs(image_val)
        os.makedirs(truth_val)
        os.makedirs(image_test)
        os.makedirs(truth_test)
        os.makedirs(scan_train_path)
        os.makedirs(scan_val_path)
        os.makedirs(scan_test_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

    patient_list = get_patients(path_to_lidc)
    random.shuffle(patient_list)

    train, val, test = train_test_val_split(patient_list)

    save_ims(train, truth_train, image_train, scan_train_path)
    save_ims(val, truth_val, image_val, scan_val_path)
    save_ims(test, truth_test, image_test, scan_test_path)


if __name__ == '__main__':
    path_lidc = '/home/rhys/Documents/datasets/LIDC-IDRI/'
    save_path = '../LIDC_examples'
    prep_config(path_lidc)
    prep_data(path_lidc, save_path)
