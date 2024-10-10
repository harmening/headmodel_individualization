#!/usr/bin/env python
import os, trimesh, numpy as np
from os.path import join as pth
import scipy.io as sio
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


def cut_mesh(bnd, cut):
    pos, tris = bnd
    assert pos.shape[0] == np.max(tris) + 1
    
    # Mark vertices below the cut value
    deleted = np.where(pos[:, 2] < cut)[0]

    # Filter out triangles containing any deleted vertices
    mask = ~np.isin(tris, deleted).any(axis=1)
    tris = tris[mask]

    # Identify unique vertices used in the remaining triangles
    tris_set = np.unique(tris)

    # Map old indices to new ones for remaining vertices
    new_pos = pos[tris_set]
    mapping = np.full(pos.shape[0], -1, dtype=int)
    mapping[tris_set] = np.arange(len(tris_set))

    # Remap triangle indices
    tris = mapping[tris]

	# Check that the new mesh is valid
    assert np.min(tris) == 0
    assert new_pos.shape[0] == np.max(tris) + 1
    
    return (new_pos, tris), deleted



if __name__ == '__main__':
    ### Load photogrammetry surface mesh cut below the ears ###
    photogrammetry_fn = pth(BASEDIR, 'data', 'photogrammetry_test_data', \
    #                        'cutscan.obj')
    #                        'cutscan.stl')
    #                        'cutscan.ply')
                            'cutscan_decimated.stl') 
    cutscalp = trimesh.load(photogrammetry_fn)
    CUT = 30#mm above the LPA, RPA, NAS plane
    cut_bnd, _ = cut_mesh((cutscalp.vertices, cutscalp.faces), CUT)
    cutscalp = trimesh.Trimesh(cut_bnd[0], cut_bnd[1])

    ### Too many points? -> Decimate mesh first ###
    # e.g. by using MNE-Python:
    """
    import mne
    vertices, faces = mne.decimate_surface(cutscalp.vertices, cutscalp.faces,
                                           n_triangles=50, method="quadric")
    decimated = trimesh.Trimesh(vertices, faces)
    cutscalp = decimated
    cutscalp.export(pth(BASEDIR, 'data', 'photogrammetry_test_data', \
                        'cutscan_decimated.stl'), file_type='stl_ascii')
    """
    scalp_proxies = np.array(cutscalp.vertices)

    
    # Input fiducials here: 
    nas = np.array([144.482786, 129.291732, 380.645666])
    rpa = np.array([80.580458, 21.190605, 362.990298])
    lpa = np.array([154.618663, 59.192534, 488.364463])
    fiducials = np.array([nas, lpa, rpa])



    ### Transform into ctf coordinate system ###
    fiducials, transform = transform_to_ctf(fiducials, nas, lpa, rpa, True)
    scalp_proxies = transform_to_ctf(scalp_proxies, nas, lpa, rpa)


    ### Call pca warping function ###
    # depending on the amount of scalp proxy points this may take a while 
    # (69 points -> ~1h)
    # (120 points -> ~3h)
    bnd_w = pca_surfacemesh_warping(fiducials, scalp_proxies)


    ### Transform back ###
    for shell in SHELLS:
        bnd_w[shell] = (apply_transform(np.linalg.pinv(transform), 
                                        bnd_w[shell][0]),
                        bnd_w[shell][1])
  

    ### save results in different formats ###
    photogrammetry_pth = os.path.dirname(photogrammetry_fn)
    for shell in SHELLS:
        write_tri(bnd_w[shell][0], bnd_w[shell][1], pth(photogrammetry_pth,
                                                        'pca_warped_%s.tri'
                                                        % shell))
        mesh = trimesh.Trimesh(vertices=bnd_w[shell][0], faces=bnd_w[shell][1]) 
        mesh.export('pca_warped_%s.stl' % shell, file_type='stl_ascii')

    sio.savemat(pth(photogrammetry_pth, 'pca_warped_bnd.mat'), {'bnd': bnd_w})
    np.save(pth(photogrammetry_pth, 'pca_warped_bnd.npy'), bnd_w)

     
    ### For plotting ###
    """
    with open(pth(photogrammetry_pth, 'scalp_proxies.txt'), 'w') as f:
        for vert in scalp_proxies:
            f.write('%f %f %f\n' % (vert[0], vert[1], vert[2]))
    """
