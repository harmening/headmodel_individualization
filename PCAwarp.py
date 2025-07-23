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
NUM_PCAS = 'ALL' #16

# Don't change the following paths
BASEDIR = os.path.dirname(os.path.realpath(__file__))
PCAS = pth(BASEDIR, 'data', 'pcas', NUM_PCAS+'pcas.npy')
MEAN_HEAD = pth(BASEDIR, 'data', 'pcas', 'mean_head.npy')
STD_DEV = pth(BASEDIR, 'data', 'pcas', 'std_dev.npy')
SHELLS = ['scalp', 'skull', 'csf', 'cortex']



def pca_surfacemesh_warping(fiducials, optodes):
    # Load PCA data
    mean_bnd = np.load(MEAN_HEAD, allow_pickle=True).item()
    std_dev = np.load(STD_DEV, allow_pickle=True)
    pcas = np.load(PCAS, allow_pickle=True)

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

    ## Python-MNE (.surf)
    for shell in SHELLS:
        try:
            import nibabel as nib
        except:
            pass
        else:
            mne_names = {'scalp': 'outer_skin.surf',
                         'skull': 'outer_skull.surf',
                         'csf': 'inner_skull.surf',
                         'cortex': 'inner_csf.surf'}
            nib.freesurfer.io.write_geometry(pth(photogrammetry_pth, mne_names[shell]),
                                             bnd_w[shell][0]/1000, bnd_w[shell][1])

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
    output_dir = pth(photogrammetry_pth, 'cedalion')
    os.makedirs(output_dir, exist_ok=True)
    back_transform = np.linalg.pinv(transform) @ ras2ctf # first back to ctf,
                                                         # then in phtgrammetry
                                                         # coordinate system
    # back_transform = np.eye(4) #for RAS
    tri2nii(bnds, output_dir=output_dir, transform=back_transform, 
            t1_fn='src/template.nii', meshes='all')
    print('Masks created. Do some postprocessing:')

    postprocessing(output_dir, num_tissues=4)
    print('Finished.')
