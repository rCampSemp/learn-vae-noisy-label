import pytest
from preprocessing.Prepare_LIDC import PrepareLIDC
import pylidc as pl
import numpy as np
import os

@pytest.fixture
def prep_lidc():
    keylist = ['patient_id','nodule_no','subtlety', 'internalStructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']

    path_lidc = '/home/rhys/Documents/datasets/LIDC-IDRI/'
    save_path = './LIDC_examples'

    lidc_process = PrepareLIDC(path_lidc, save_path, clevel=0.5, mask_threshold=30, resolution=64, meta_keys=keylist)
    return lidc_process

def test_config(prep_lidc):
    prep_lidc.config()
    
    with open('/home/rhys/.pylidcrc') as testfile:
        assert f'[dicom]\npath = {prep_lidc.LIDC_path}\nwarn = True\n\n' == testfile.read()
    
    try:
        os.remove('/home/rhys/.pylidcrc')
    except OSError as error:
        print(error)

def test_make_folders(prep_lidc, tmpdir):
    prep_lidc.save_path = tmpdir
    prep_lidc.make_folders()
    
    for split in ['train', 'test', 'validate']:
        assert os.path.exists(prep_lidc.save_path + '/' + split + '/masks/annots')
        assert os.path.exists(prep_lidc.save_path + '/' + split + '/masks/GT')
        assert os.path.exists(prep_lidc.save_path + '/' + split + '/scans')

    assert os.path.exists(prep_lidc.save_path + '/meta')

def test_reshape(prep_lidc):
    test_bbox = (slice(120, 300, None), slice(30, 99, None), slice(12, 24, None))
    actual_bbox = prep_lidc.reshape(test_bbox)
    expected_bbox = (slice(178, 242, None), slice(32, 96, None), slice(12, 24, None))
    assert actual_bbox == expected_bbox

def test_getAverageParams(prep_lidc):
    scan = pl.query(pl.Scan).first()
    nodules = scan.cluster_annotations()
    nod2 = nodules[1]

    actual_av_metrics = prep_lidc.getAverageParams(nod2)
    expected_av_metrics = [5, 1, 6, 4, 3, 3, 2, 4, 4]

    assert actual_av_metrics == expected_av_metrics

def test_standardize(prep_lidc):
    img_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    actual_std_img = prep_lidc._standardize(img_array)
    expected_std_img = np.array([[-1.34164079, -0.4472136 ], [ 0.4472136 ,  1.34164079]])

    assert np.allclose(actual_std_img, expected_std_img, atol=1e-10)

