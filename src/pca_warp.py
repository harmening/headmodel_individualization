#!/usr/bin/env python
import os, numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import trimesh, tempfile
from random import random 
import src.pca_pycaster as pycaster
from numba import jit

# only for triangle difference regularization (compile cython code before):
# https://github.com/MattiaMontanari/openGJK
#import src.openGJK_cython as opengjk 



def elec_warp(elecpos, pcas, mean_head, std_dev, onebyone=False):
    # Warp to electrode positions or other point clouds
    scalp_tris = mean_head['scalp'][1]
    all_tris = {k: v[1] for k, v in mean_head.items()}
    assert (scalp_tris == mean_head['cortex'][1]).all()
    num_pcas, pca_dim, dim = pcas.shape
    shells = list(mean_head.keys())
    bndsize = [len(mean_head[shell][0]) for shell in shells]
    # Determine mean_pnt for surface-line-intersection 
    mean_pnt = np.mean(mean_head['cortex'][0], axis=0)
    min_idx, min_dist = shortest_dist(mean_pnt, mean_head['scalp'][0])
    new_z = np.max(mean_head['scalp'][0][:,2]) - min_dist
    mean_pnt[2] = new_z

    # Minimize shape_distance
    if not onebyone:
        # all PCs at the same time 
        x0 = np.ones(num_pcas)
        res = minimize(error, x0, args=(pcas, mean_head, std_dev, all_tris, \
                                        mean_pnt, elecpos),
                                  options={'disp':True,'eps':0.5}) # mm
        print(res)
        x_p = res.x
   
    else:
        # Minimize 1 PC after another
        x_p = np.zeros(num_pcas)
        for i in range(num_pcas):
            x0 = 1.0
            res = minimize(error1by1, x0, args=(i, x_p, pcas, mean_head, \
                                                std_dev, all_tris, mean_pnt, \
                                                elecpos),
                                        options={'disp':True,'eps':0.1})
            print("%d. PCA coefficient: %f" % (i+1, res.x[0]))
            x_p[i] = res.x[0]
    
    pos = pca2tri(x_p, pcas, mean_head, std_dev, wise='head')
    reconstructed = {shell: (pos[shell], mean_head[shell][1]) \
                     for shell in shells}
    return reconstructed


def pca_warp(head, pcas, mean_bnd, std_dev, known='scalp'):
    # Warp to known vertex positions of SAME TRIANGULATION as usied in PCA
    # construction
    shells = head.keys()
    idx = [i for i, shell in enumerate(shells) if shell==known][0]
    num_pcas, dim_pcas, dim = pcas.shape
    known_bnd = head[known][0]
    num_pos, dim = known_bnd.shape

    # flatten vertex coordinates and standardize
    tot = num_pos * dim
    known_bnd = ((known_bnd.reshape(tot) - mean_bnd[idx*tot:(idx+1)*tot])  \
                 / std_dev[idx*tot:(idx+1)*tot])
    known_bnd.reshape((num_pos, dim))
    bndsize = [len(head[shell][0]) for shell in shells]
    start = sum(bndsize[:idx])
    size = bndsize[idx]
    assert size == num_pos
    # prepare minimization
    X = pcas[:,start:start+size,:].reshape(num_pcas, size*dim).T
    y = known_bnd.reshape((num_pos*dim))
    # do fit until sufficient accuracy is achieved
    reg = LinearRegression(fit_intercept=True, normalize=False).fit(X, y)
    repeat = 0
    while reg.score(X, y) < 0.1 and repeat < 5:
        reg = LinearRegression(fit_intercept=True, normalize=False).fit(X, y)
        repeat += 1
    #print('Score of lin.reg warping: %f' % reg.score(X, y))
    x_p = reg.coef_
    # Reconstruct surface meshes from PC parameters
    warped = {}
    for s, shell in enumerate(shells):
        start = sum(bndsize[:s])
        size = bndsize[s]
        warped[shell] = pcas[:,start:start+size,:].reshape(num_pcas, size*dim)
        warped[shell] = warped[shell].T.dot(x_p)
        start *= dim  
        size *= dim  
        warped[shell] *= std_dev[start:start+size]
        warped[shell] += mean_bnd[start:start+size]
        warped[shell] = warped[shell].reshape(bndsize[s], dim)
    return warped


def caster(pos, tris):
    mesh = trimesh.Trimesh(pos, tris)
    fn = os.path.join(tempfile.mkdtemp(), 'stl_'+str(random()))
    mesh.export(file_obj=fn, file_type='stl_ascii')
    vtk_caster = pycaster.rayCaster.fromSTL(fn, scale=1.0)  
    os.remove(fn)
    return vtk_caster

def shortest_dist(vert_pos, list_of_pos):
    min_idx, min_dist = 0, np.linalg.norm(list_of_pos[0] - vert_pos)
    dist = np.linalg.norm(np.array(list_of_pos) - vert_pos, axis=1)  
    min_idx = np.argmin(dist)
    min_dist = dist[min_idx]
    return min_idx, min_dist


def shape_distance(elecpos, bnd, mean_pnt):
    pos, tris = bnd
    fit = caster(pos, tris)
    proj = elecpos + 1000*(elecpos-mean_pnt)
    intsct = []
    for i, elec in enumerate(elecpos):
        # find intersection with mesh
        intersections = fit.castRay(mean_pnt, proj[i])
        if len(intersections) == 1:
            min_idx = 0
        elif len(intersections) > 1:
            print('More than one intersection!')
            min_idx, _ = shortest_dist(elec, intersections)
        else:
            print('Zero intersections!')
            intersections = pos
            min_idx, _ = shortest_dist(elec, intersections)
        intsct.append(intersections[min_idx])
    dist = np.linalg.norm(elecpos - np.array(intsct))
    diff = np.mean(dist)
    return diff


def pca2tri(coeff, pcas, mean_head, std_dev, wise='head'):
    shells = list(mean_head.keys())
    num_pcas, dim_pcas, dim = pcas.shape
    reconstructed = {}
    bndsize = [len(mean_head[shell][0]) for shell in shells]
    mean_bnd = np.zeros((sum(bndsize)*dim))
    for s, shell in enumerate(shells):
        size = bndsize[s]*dim
        start = sum(bndsize[:s])*dim
        mean_bnd[start:start+size] = mean_head[shell][0].flatten()
    reshaped_mean_bnd = []
    reshaped_std_dev = []
    for s, shell in enumerate(shells):
        size = bndsize[s]*dim
        start = sum(bndsize[:s])*dim
        reshaped_mean_bnd.append(
                mean_bnd[start:start+size].reshape(bndsize[s], dim))
        reshaped_std_dev.append(
                std_dev[start:start+size].reshape(bndsize[s], dim))
    if wise == 'bnd':
        for s, shell in enumerate(shells):
            size = bndsize[s]
            start = sum(bndsize[:s])
            X = pcas[:,start:start+size,:].reshape(num_pcas, size*dim).T
            reconstructed[shell] = (X.dot(coeff[shell]) * reshaped_std_dev[s])\
                                   + reshaped_mean_bnd[s]
            reconstructed[shell] = reconstructed[shell].reshape(size, dim)
    elif wise == 'head': 
        X = pcas.reshape((num_pcas, sum(bndsize)*dim)).T 
        reconstructed_all = (X.dot(coeff) * std_dev) + mean_bnd
        for s, shell in enumerate(shells):
            size = bndsize[s]*dim
            start = sum(bndsize[:s])*dim
            reconstructed[shell] = \
                    reconstructed_all[start:start+size].reshape(bndsize[s],
                                                                dim)
    else:
        raise ValueError
    return reconstructed


def error(x_p, pcas, mean_head, std_dev, all_tris, mean_pnt, elecpos, 
          regularize=False):
    pos = pca2tri(x_p, pcas, mean_head, std_dev, wise='head')
    scalp_tris = all_tris['scalp']
    fit_bnd = (pos['scalp'], scalp_tris)
    diff = shape_distance(elecpos, fit_bnd, mean_pnt)

    # regularize mesh intersections (vertex difference) # best option
    if regularize:
        shells = list(mean_head.keys())
        shells = ['cortex', 'csf', 'skull', 'scalp']
        bndsize = np.array([len(pos[shell]) for shell in shells])
        warped = np.zeros((np.sum(bndsize), 3))
        start = 0
        for s, shell in enumerate(shells):
            warped[start:start+bndsize[s],:] = pos[shell]
            start += bndsize[s]
        diff += regularizer(warped, bndsize)

    """
    ## regularize mesh intersections (triangle difference) # slow!
    if regularize: 
        shells = ['cortex', 'csf', 'skull', 'scalp']
        warped = {shell: (pos[shell], all_tris[shell]) for shell in shells}
        triangles = {shell: np.array([[warped[shell][0][i] for i in t] \
                     for t in warped[shell][1]]) for shell in shells}
        for s in range(1, len(shells)):
            #for ii in triangles[shells[s-1]]:
            #    for jj in triangles[shells[s]]:
            #        dist = opengjk.pygjk(ii, jj)
            #        if dist < 0.001:
            #            diff += (0.001-dist)
            diff += opengjk.pygjk_penalize_close_meshes(triangles[shells[s-1]],
                                                        triangles[shells[s]],
                                                        thresh=5.0) #5mm

    # regularize edge length  # not so successful
    if regularize:
        for shell in ['skull', 'csf', 'cortex']:
            pdist = [[pos[shell][p] for p in face] for face in scalp_tris]
            edge_len = [[abs(np.linalg.norm(p[0]-p[1])), 
                         abs(np.linalg.norm(p[0]-p[2])),
                         abs(np.linalg.norm(p[1]-p[2]))] for p in pdist]
            pdist = [abs(e[0]-e[1]) + abs(e[1]-e[2]) + abs(e[0]-e[2]) \
                     for e in edge_len]
            diff += np.mean(np.array(pdist).flatten())  

    """
    return diff 


@jit(nopython=True, parallel=True)
def regularizer(warped, bndsize):
    diff = 0
    start = 0
    for s in range(0, len(bndsize)-1):
        pts1 = warped[start:start+bndsize[s]]
        start += bndsize[s]
        pts2 = warped[start:start+bndsize[s+1]]

        # reshaping to be able to calculate the distance matrix
        a_reshaped = pts1.reshape(pts1.shape[0], 1, 3)
        b_reshaped = pts2.reshape(1, pts2.shape[0], 3)

        #calculation of all distances between all points
        norms = np.sqrt(np.sum((a_reshaped - b_reshaped)**2, axis=2)).flatten()

        # penalize too short distances
        #penalty = (0.010-norms[norms < 0.010])
        penalty = (5.0-norms[norms < 5.0]) #5mm
        diff += np.sum(penalty)
    return diff




def error1by1(x_i, i, x_p, pcas, mean_head, std_dev, all_tris, mean_pnt, elecpos):
    x_p[i] = x_i
    return error(x_p, pcas, mean_head, std_dev, all_tris, mean_pnt, elecpos)
