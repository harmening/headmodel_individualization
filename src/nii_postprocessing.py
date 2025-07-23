#!/usr/bin/env python
import os
from os.path import join as pth
from shutil import copyfile
import nibabel as nib
import gzip, shutil
from src.cedalion_geometry_segmentation import segmentation_postprocessing as \
        cgeoseg_seg_postproc

def postprocessing(seg_datadir, num_tissues=4):
    if num_tissues == 6:
        # Current order: 'air', 'scalp', 'skull', 'csf', 'gm', 'wm'
        # postprocessing expects: 'gm', 'wm', 'csf', 'bone', 'skin', 'air'
        shells = ['air', 'scalp', 'skull', 'csf', 'gm', 'wm']
        mask_files = {
            "gray":  "mask_5.nii",
            "white": "mask_6.nii",
            "csf":   "mask_4.nii",
            "bone":  "mask_3.nii",
            "skin":  "mask_2.nii",
            "air":   "mask_1.nii",
        }
    elif num_tissues == 5:
        shells = ['air', 'scalp', 'skull', 'csf', 'whitegray']
        mask_files = {
            "whitegray": "mask_5.nii",
            "csf":       "mask_4.nii",
            "bone":      "mask_3.nii",
            "skin":      "mask_2.nii",
            "air":       "mask_1.nii",
        }
    elif num_tissues == 4:
        # Adding here "air mask" is a little trick to be able to use
        # the labelUnassigned option of the postprocessing since the tri2nii is
        # not working perfectly and there are a lot of unassigned voxels
        shells = ['air', 'scalp', 'skull', 'csf', 'cortex']
        mask_files = {
            "cortex": "mask_4.nii",
            "csf":    "mask_3.nii",
            "bone":   "mask_2.nii",
            "skin":   "mask_1.nii",
            "air":    "mask_0.nii",
        }
    

    for i in range(num_tissues):
        with gzip.open(pth(seg_datadir, ''.join(['mask_', str(i+1),
                                                 '.nii.gz'])), 'rb') as f_in:
            with open(pth(seg_datadir, ''.join(['mask_', str(i+1),
                                                '.nii'])), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    

    if num_tissues == 4:
        # create an empty air mask
        mask_1 = nib.load(pth(seg_datadir, 'mask_1.nii'))
        dat = mask_1.get_fdata()
        dat[:,:,:] = 1.0 # set all to zero 
        mask_0 = nib.Nifti1Image(dat, mask_1.affine, mask_1.header)  
        nib.save(mask_0, pth(seg_datadir, 'mask_0.nii'))
   


    removeAir = False if num_tissues == 4 else True
    removeAir = True
    labelUnassigned = True if num_tissues == 4 else False
    mask_files = cgeoseg_seg_postproc(seg_datadir,
                                      mask_files,
                                      isSmooth=False,
                                      fixCSF=True,
                                      removeDisconnected=True, 
                                      labelUnassigned=labelUnassigned,
                                      removeAir=removeAir,
                                      subtractTissues=True)
    

    if num_tissues == 4:
        os.remove(pth(seg_datadir, 'mask_air.nii'))
