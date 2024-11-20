#!/usr/bin/env python
import os, argparse
from os.path import join as pth
import trimesh, numpy as np, scipy.io as sio
from src.tri_io import write_tri, load_tri
from src.transform_to_ctf import transform_to_ctf, apply_transform
from src.pca_warp import shortest_dist, elec_warp

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
                        type=float, nargs=3, required=True)
    parser.add_argument('-lpa', help='Left preauricular (LPA) fiducial \
                        coordinates.', type=float, nargs=3, required=True)
    parser.add_argument('-rpa', help='Right preauricular (RPA) fiducial \
                        coordinates.', type=float, nargs=3, required=True)

    ### Parse arguments
    scalp = parser.parse_args().scalp
    nas = np.array(parser.parse_args().nas)
    lpa = np.array(parser.parse_args().lpa)
    rpa = np.array(parser.parse_args().rpa)


    ### Load scalp proxy / photogrammetry mesh / electrodes
    if scalp.endswith('.npy'):
        scalp_proxies = np.load(scalp)
    elif scalp.endswith('.txt'):
        scalp_proxies = np.loadtxt(scalp)
    else:
        scalp_proxies = np.array(trimesh.load(scalp).vertices)
   

    ### Transform into ctf coordinate system
    scalp_proxies, transform = transform_to_ctf(scalp_proxies, nas, lpa, rpa,
                                                return_transform=True)


    ### Cut mesh input points above the ears
    CUT = 30#mm above the LPA, RPA, NAS plane
    scalp_proxies = scalp_proxies[scalp_proxies[:, 2] > CUT]


    ### Too many points? -> Decimate 
    if len(scalp_proxies) > 50:
        idx = np.random.randint(len(scalp_proxies), size=50)
        scalp_proxies = scalp_proxies[idx,:]
   

    ### PCA warping
    fiducials = np.array([nas, lpa, rpa])
    bnd_w = pca_surfacemesh_warping(fiducials, scalp_proxies)


    ### Transform back
    for shell in SHELLS:
        bnd_w[shell] = (apply_transform(np.linalg.pinv(transform), 
                                        bnd_w[shell][0]),
                        bnd_w[shell][1])
  

    ### Save results in different formats
    photogrammetry_pth = os.path.dirname(scalp)
    for shell in SHELLS:
        write_tri(bnd_w[shell][0], bnd_w[shell][1], pth(photogrammetry_pth,
                                                        'pca_warped_%s.tri'
                                                        % shell))
        mesh = trimesh.Trimesh(vertices=bnd_w[shell][0], faces=bnd_w[shell][1]) 
        mesh.export('pca_warped_%s.stl' % shell, file_type='stl_ascii')

    sio.savemat(pth(photogrammetry_pth, 'pca_warped_bnd.mat'), {'bnd': bnd_w})
    np.save(pth(photogrammetry_pth, 'pca_warped_bnd.npy'), bnd_w)
