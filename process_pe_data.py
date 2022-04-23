import pandas as pd 
import nibabel as nib
import pydicom
import tqdm
import datetime, time
import numpy as np
import cv2

from pathlib import Path
from collections import defaultdict
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ImplicitVRLittleEndian


ANNOTATIONS_DIR = Path('/local-scratch-nvme/nigam/projects/PE/intermountain/marked/all')
DICOM_DIR = Path('/local-scratch-nvme/nigam/projects/PE/intermountain/')
INTERMOUNTAIN_META_CSV = Path('./intermountain_meta.csv')
DEST_PATH = Path('/local-scratch-nvme/nigam/projects/PE/intermountain/marked/dicom')
INTERMOUNTAIN_LABEL_CSV = Path('./intermountain_label.csv')
RSNA_META_CSV = Path('/local-scratch-nvme/nigam/projects/PE/rsna/rsna/rsna_train_master.csv')
RSNA_LABEL_CSV = Path('./rsna_label.csv')
RSNA_PATH = Path('/local-scratch-nvme/nigam/projects/PE/rsna/rsna/train')

metadata = defaultdict(list)

def get_metadata():
    if INTERMOUNTAIN_META_CSV.exists():
        metadata_df = pd.read_csv(INTERMOUNTAIN_META_CSV)
    else: 
        # get metadata
        for pth in DICOM_DIR.iterdir(): 
            if 'marked' in str(pth): continue
            for f in pth.iterdir():
                dcm = pydicom.dcmread(str(f))
                metadata['patient_name'].append(str(dcm.PatientName).replace('^','_'))
                try: 
                    metadata['patient_position'].append(dcm.ImagePositionPatient[-1])
                except: 
                    metadata['patient_position'].append(0)
                try: 
                    metadata['slice_thickness'].append(dcm.SliceThickness)
                except: 
                    metadata['slice_thickness'].append(0)
        metadata_df = pd.DataFrame.from_dict(metadata)
        metadata_df['reviewed'] = False
        metadata_df['label'] = 0 
        metadata_df.to_csv(str(INTERMOUNTAIN_META_CSV))
    return metadata_df

def read_annotations(): 

    annotation_dfs = []
    metadata = defaultdict(list)

    # loop over annotations
    for pth in ANNOTATIONS_DIR.iterdir(): 
        for f in pth.iterdir(): 
            patient_name = str(f).split('/')[-2]
            if 'Segmentation' in str(f):
                print(f)
                labels = nib.load(str(f)).get_fdata()
                n_slices = labels.shape[2]
            else: 
                images = nib.load(str(f)).get_fdata()


        for idx in range(n_slices): 
            label = labels[:,:,idx]
            pixel_array = images[:,:,idx]

            metadata['patient_name'].append(patient_name)
            if label[label == 1].any(): 
                metadata['label'].append(1)
            else: 
                metadata['label'].append(0)

            filename = DEST_PATH / f"{patient_name}_{idx}.npy"
            pixel_array = cv2.resize(pixel_array, (256, 256), interpolation=cv2.INTER_AREA)
            np.save(filename, pixel_array)

            metadata['path'].append(filename)


            #patient_df = meta_df[meta_df['patient_name'] == patient_name.upper()]
            #thickness_count = {v:k for k,v in patient_df['slice_thickness'].value_counts().to_dict().items()}
            #thickness = thickness_count[n_slices]
            #patient_df = patient_df[patient_df['slice_thickness'] == thickness] 
    metadata_df = pd.DataFrame.from_dict(metadata)
    metadata_df.to_csv(INTERMOUNTAIN_LABEL_CSV)

def parse_rsna(): 
    metadata = defaultdict(list)
    df = pd.read_csv(RSNA_META_CSV)
    #pos_df = df[(df.Institution == 'Stanford') & (df.negative_exam_for_pe == 0)]
    pos_df = df[(df.negative_exam_for_pe == 0)]

    for idx, row in pos_df.iterrows(): 
        filename = DEST_PATH / f"{row['StudyInstanceUID']}_{row['SOPInstanceUID']}.npy"

        metadata['label'].append(row['pe_present_on_image'])
        metadata['patient_name'].append(row['StudyInstanceUID'])
        metadata['path'].append(filename)
        metadata['institution'].append(row['Institution'])

        dcm = pydicom.dcmread(RSNA_PATH / row['InstancePath'])
        pixel_array = dcm.pixel_array
        intercept = dcm.RescaleIntercept
        slope = dcm.RescaleSlope
        pixel_array = pixel_array * slope + intercept


        pixel_array = cv2.resize(pixel_array, (256, 256), interpolation=cv2.INTER_AREA)
        np.save(filename, pixel_array)

    metadata_df = pd.DataFrame.from_dict(metadata)
    metadata_df.to_csv(RSNA_LABEL_CSV)

if __name__ == "__main__": 
    #metadata_df = get_metadata() 
    #read_annotations()
    #parse_rsna() 
    rsna_df = pd.read_csv(RSNA_LABEL_CSV)
    inter_df = pd.read_csv(INTERMOUNTAIN_LABEL_CSV)
    df = pd.concat([rsna_df, inter_df])
    df = df[['label', 'patient_name', 'path', 'institution']]

    df.to_csv('ssl_fl_pe.csv')

