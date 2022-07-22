import pytest
from preprocessing.Prepare_LIDC import PrepareLIDC
import pylidc as pl

@pytest.fixture
def prep_lidc():
    keylist = ['patient_id','nodule_no','subtlety', 'internalStructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']

    path_lidc = '/home/rhys/Documents/datasets/LIDC-IDRI/'
    save_path = '../LIDC_examples'

    lidc_process = PrepareLIDC(path_lidc, save_path, clevel=0.5, mask_threshold=30, resolution=64, meta_keys=keylist)
    return lidc_process

def test_config(prep_lidc):
    prep_lidc.config()
    
    with open('/home/rhys/.pylidcrc') as testfile:
        assert f'[dicom]\npath = {prep_lidc.LIDC_path}\nwarn = True\n\n' == testfile.read()

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

