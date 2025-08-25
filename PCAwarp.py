#!/usr/bin/env python
import os, argparse
from os.path import join as pth
import numpy as np, scipy.io as sio
from src.tri_io import write_tri, load_tri
from src.transform_to_ctf import transform_to_ctf, apply_transform
from src.pca_warp import shortest_dist, elec_warp
from src.tri2nii import tri2nii
from src.nii_postprocessing import postprocessing



# How many PCs to use for reconstruction?
NUM_PCAS = 50 # out of 316

# Don't change the following paths
BASEDIR = os.path.dirname(os.path.realpath(__file__))
PCAS = pth(BASEDIR, 'data', 'pcas', 'ALLpcas.npy')
MEAN_HEAD = pth(BASEDIR, 'data', 'pcas', 'mean_head.npy')
STD_DEV = pth(BASEDIR, 'data', 'pcas', 'std_dev.npy')
SHELLS = ['scalp', 'skull', 'csf', 'cortex']



def pca_surfacemesh_warping(fiducials, optodes):
    # Load PCA data
    mean_bnd = np.load(MEAN_HEAD, allow_pickle=True).item()
    std_dev = np.load(STD_DEV, allow_pickle=True)
    pcas = np.load(PCAS, allow_pickle=True)
    if len(pcas) < NUM_PCAS:
        raise ValueError('Asked for too much PCs. Check the NUM_PCAS argument.')
    pcas = pcas[:NUM_PCAS]

    # Determine mean_pnt for surface-line-intersection
    mean_pnt = np.mean(mean_bnd['cortex'][0], axis=0)
    min_idx, min_dist = shortest_dist(mean_pnt, mean_bnd['scalp'][0])
    new_z = np.max(mean_bnd['scalp'][0][:,2]) - min_dist
    mean_pnt[2] = new_z

    # Reconstruction from scanned scalp
    reconstructed = elec_warp(optodes, pcas, mean_bnd, std_dev)

    return reconstructed



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-scalp', help='Path to scalp proxy file. Can be a \
                        .npy- or a .txt-file or a surface mesh of typical \
                        mesh formats like .stl, .obj, .ply', type=str,
                        required=True)
    parser.add_argument('-nas', help='Nasion (NAS) fiducial coordinates.',
                        type=float, nargs=3, default=None)
    parser.add_argument('-lpa', help='Left preauricular (LPA) fiducial \
                        coordinates.', type=float, nargs=3, default=None)
    parser.add_argument('-rpa', help='Right preauricular (RPA) fiducial \
                        coordinates.', type=float, nargs=3, default=None)

    ### Parse arguments
    scalp = parser.parse_args().scalp
    nas = np.array(parser.parse_args().nas)
    lpa = np.array(parser.parse_args().lpa)
    rpa = np.array(parser.parse_args().rpa)


    ### Load scalp proxy / photogrammetry mesh / electrodes
    print('Load input data.')
    if scalp.endswith('.npy'):
        scalp_proxies = np.load(scalp)
    elif scalp.endswith('.txt'):
        scalp_proxies = np.loadtxt(scalp)
    elif scalp.endswith('.bvct'):
        # CapTrak .bvct file
        import mne
        captrak = mne.channels.read_dig_captrak(scalp)
        # Extract EEG channel locations and fiducials
        channels = np.array([dig['r'] for dig in captrak.dig if dig['kind'] ==
                             mne.io.constants.FIFF.FIFFV_POINT_EEG])
        scalp_proxies = channels * 1000 # m -> mm
        fiducials = [dig for dig in captrak.dig if dig['kind'] ==
                     mne.io.constants.FIFF.FIFFV_POINT_CARDINAL]
        for dig in fiducials:
            if dig['ident'] == mne.io.constants.FIFF.FIFFV_POINT_NASION:
                nas = dig['r'] * 1000
            elif dig['ident'] == mne.io.constants.FIFF.FIFFV_POINT_LPA:
                lpa = dig['r'] * 1000
            elif dig['ident'] == mne.io.constants.FIFF.FIFFV_POINT_RPA:
                rpa = dig['r'] * 1000
            else:
                pass
    elif scalp.endswith('.hsp') or scalp.endswith('.elp') or scalp.endswith('.eeg'):
        # Polhemus Fastrak files
        import mne
        polhemus = mne.channels.read_dig_polhemus_isotrak(scalp)
        # Extract EEG channel locations and fiducials
        channels = np.array([dig['r'] for dig in polhemus.dig if dig['kind'] ==
                             mne.io.constants.FIFF.FIFFV_POINT_EEG])
        scalp_proxies = channels * 1000 # m -> mm
        fiducials = [dig for dig in polhemus.dig if dig['kind'] ==
                     mne.io.constants.FIFF.FIFFV_POINT_CARDINAL]
        for dig in fiducials:
            if dig['ident'] == mne.io.constants.FIFF.FIFFV_POINT_NASION:
                nas = dig['r'] * 1000
            elif dig['ident'] == mne.io.constants.FIFF.FIFFV_POINT_LPA:
                lpa = dig['r'] * 1000
            elif dig['ident'] == mne.io.constants.FIFF.FIFFV_POINT_RPA:
                rpa = dig['r'] * 1000
            else:
                pass
    elif scalp.endswith('.elc'):
        # e.g. from Polaris Vicra
        chs, coords = [], []
        with open(scalp, 'r') as f:
            for line in f:
                if ':' in line:
                    parts = line.strip().split(':')
                    label = parts[0].strip()
                    xyz = list(map(float, parts[1].strip().split()))
                    if label in ['NAS', 'Nz', 'Nasion']:
                        nas = np.array(xyz)
                    elif label in ['LPA', 'Lpa', 'LeftEar']:
                        lpa = np.array(xyz)
                    elif label in ['RPA', 'Rpa', 'RightEar']:
                        rpa = np.array(xyz)
                    else:
                        chs.append(label)
                        coords.append(xyz)
        scalp_proxies = np.array(coords)
    else:
        # Load mesh
        try:
            import trimesh
            scalp_proxies = np.array(trimesh.load(scalp).vertices)
        except:
            print('Could not load scalp file. Please check the file format.')
            raise

    ### Check if fiducials are provided   np.eye(4),#
    if nas is None or lpa is None or rpa is None:
        print('Please provide fiducial coordinates (nas, lpa, rpa).')
        raise


    ### Transform into ctf coordinate system
    mean_scalp = np.load(MEAN_HEAD, allow_pickle=True).item()['scalp'][0]
    scalp_proxies, transform = transform_to_ctf(scalp_proxies, nas, lpa, rpa,
                                                mean_scalp=mean_scalp,
                                                return_transform=True)

    ### Cut mesh input points above the ears
    CUT = 30#mm above the LPA, RPA, NAS plane
    scalp_proxies = scalp_proxies[scalp_proxies[:, 2] > CUT]


    ### Too many points? -> Decimate
    if len(scalp_proxies) > 100:
        idx = np.random.randint(len(scalp_proxies), size=100)
        scalp_proxies = scalp_proxies[idx,:]


    ### PCA warping
    print('Do the PCA warping. This might take a while...')
    fiducials = np.array([nas, lpa, rpa])
    bnd_w = pca_surfacemesh_warping(fiducials, scalp_proxies)



    ### Transform back
    for shell in SHELLS:
        bnd_w[shell] = (apply_transform(np.linalg.pinv(transform),
                                        bnd_w[shell][0]),
                        bnd_w[shell][1])




    ### Save results in different formats
    print('Finished. Export results in different formats.')
    photogrammetry_pth = os.path.dirname(scalp)

    ## OpenMEEG file format (.tri)
    for shell in SHELLS:
        write_tri(bnd_w[shell][0], bnd_w[shell][1], pth(photogrammetry_pth,
                                                        'pca_warped_%s.tri'
                                                        % shell))
    ## .stl
    for shell in SHELLS:
        try:
            import trimesh
        except:
            pass
        else:
            mesh = trimesh.Trimesh(bnd_w[shell][0], bnd_w[shell][1])
            mesh.export(pth(photogrammetry_pth, 'pca_warped_%s.stl' % shell),
                        file_type='stl_ascii')

    ## fieldtrip (MatLab struct .mat)
    sio.savemat(pth(photogrammetry_pth, 'pca_warped_bnd.mat'), {'bnd': bnd_w})

    ## .npy
    np.save(pth(photogrammetry_pth, 'pca_warped_bnd.npy'), bnd_w)
    np.save(pth(photogrammetry_pth, 'pca_warped_bnd_transform.npy'), transform)



    ## Cedalion (.nii), segmentation masks
    # Transform again into ctf and then into RAS (this is necessary for tri2nii)
    print('Start cedalion export.')
    ras2ctf = np.load('src/transform_acpc2ctf_icbm.npy', allow_pickle=True)
    ctf2ras = np.linalg.pinv(ras2ctf)
    for shell in SHELLS:
        bnd_w[shell] = (apply_transform(transform, bnd_w[shell][0]),
                        bnd_w[shell][1])
        bnd_w[shell] = (apply_transform(ctf2ras, bnd_w[shell][0]/1000)*1000,
                        bnd_w[shell][1])
    bnds = [(bnd_w[shell][0], bnd_w[shell][1]) for shell in SHELLS] # mm
    cedalion_output_dir = pth(photogrammetry_pth, 'cedalion')
    os.makedirs(cedalion_output_dir, exist_ok=True)
    #back_transform = np.linalg.pinv(transform) @ ras2ctf # first back to ctf,
    #                                                     # then in phtgrammetry
    #                                                     # coordinate system
    back_transform = np.eye(4) #for RAS
    tri2nii(bnds, output_dir=cedalion_output_dir, transform=back_transform,
            t1_fn='src/template.nii', meshes='all')
    print('Masks created. Do some postprocessing:')
    postprocessing(cedalion_output_dir, num_tissues=4)
    # Clean up masks
    for i in range(5):
        for fn in ['mask_%d.nii' % i, 'mask_%d.nii.gz' % i]:
            if os.path.exists(pth(cedalion_output_dir, fn)):
                os.remove(pth(cedalion_output_dir, fn))

    ditigized2ras = np.linalg.pinv(ras2ctf) @ transform # first to ctf,
                                                        # then ctf2ras
    import xarray as xr
    from_crs, to_crs = "digitized", "ras"
    ditigized2ras = xr.DataArray(ditigized2ras, dims=[to_crs, from_crs])
    ditigized2ras.to_netcdf(pth(cedalion_output_dir, "t_ditigized2ras.nc"))


    ## Python-MNE (.surf)
    import nibabel as nib
    mne_output_dir = pth(photogrammetry_pth, 'mne', 'pcawarp')
    os.makedirs(pth(mne_output_dir, 'bem'), exist_ok=True)
    os.makedirs(pth(mne_output_dir, 'mri'), exist_ok=True)
    mne_names = {'scalp': 'outer_skin.surf',
                 'skull': 'outer_skull.surf',
                 'csf': 'inner_skull.surf',
                 'cortex': 'inner_csf.surf'}
    for shell in SHELLS:
        nib.freesurfer.io.write_geometry(pth(mne_output_dir, 'bem',
                                             mne_names[shell]),
                                         bnd_w[shell][0]/1000, bnd_w[shell][1])
    # + fake T1 MRI (for mne plotting)
    data = nib.load(pth(cedalion_output_dir, 'mask_skin.nii')).get_fdata()
    new_data = np.zeros(data.shape)
    tissue_color = {'skin': 0.45, # ~0.4–0.6 (depends on fat content)
                    'bone': 0.05, # ~0.0–0.1 (very dark, almost no signal)
                    'csf': 0.1, #CSF ~0.0–0.1 (dark; long T1 -> low signal)
                    'cortex': 0.55} #cortex (0.5–0.7 (wm brighter than gm)
    for lab, tissue in enumerate(['skin', 'bone', 'csf', 'cortex']):
        mask = nib.load(pth(cedalion_output_dir, f"mask_{tissue}.nii"))
        new_data[mask.get_fdata() == 1] = tissue_color[tissue]
    new_img = nib.nifti1.Nifti1Image(new_data, mask.affine, mask.header)
    nib.save(new_img, pth(mne_output_dir, 'mri', 'T1.mgz'))
    # + transform object for coregistration
    import mne
    from mne.transforms import Transform
    trans = Transform('head', 'mri', ditigized2ras)
    mne.write_trans(pth(mne_output_dir, 'ditigized2ras-trans.fif'), trans)

    print('Finished.')
