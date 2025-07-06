import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

def transform_to_ctf(pts, nas, lpa, rpa, mean_scalp=None, return_transform=False):
    transform = get_transform(nas, lpa, rpa)
    pts_ctf = apply_transform(transform, pts)
    # to minimize error due to individual fiducial picking, we first shift the
    # points to the mean head position of the PCA
    # this only works if you have the scalp proxy points evenyly scattered over
    # the whole head surface!
    if mean_scalp is not None:
        mean_scalp = mean_scalp[mean_scalp[:, 2] > 0]
        shift, min_distance = find_optimal_shift(pts_ctf, mean_scalp)
        #shift = np.array([shift[0],0,0])
        shift[1] = 0
        shift[2] = 0
        transform[0:3,3] += shift
    pts_ctf = apply_transform(transform, pts)
    if return_transform:
        return pts_ctf, transform
    return pts_ctf


def compute_total_distance(shift, A, B_tree):
    # Apply shift in x, y, z directions
    A_shifted = A + shift  # shift is a length-3 array [dx, dy, dz]

    # Find nearest neighbor in B for each point in A
    distances, _ = B_tree.query(A_shifted)

    # Return total Euclidean distance
    return np.sum(distances)

def find_optimal_shift(A, B, initial_guess=(0.0, 0.0, 0.0)):
    # Create KDTree for fast nearest neighbor search
    B_tree = cKDTree(B)

    # Minimize the distance function with respect to 3D shift
    result = minimize(compute_total_distance, initial_guess, args=(A, B_tree), method='L-BFGS-B')

    return result.x, result.fun


def get_transform(nas, lpa, rpa):

    # origin is between lpa and rpa
    origin = (lpa + rpa) / 2
    # from this compute x, y, z coordinate directions
    dirx = nas - origin
    dirz = np.cross(dirx, lpa - rpa)
    diry = np.cross(dirz, dirx)

    # normalize vectors
    dirx /= np.linalg.norm(dirx)
    diry /= np.linalg.norm(diry)
    dirz /= np.linalg.norm(dirz)

    # compute the rotation matrix
    rot = np.eye(4)
    rot[:3, :3] = np.linalg.inv(np.stack((dirx, diry, dirz)).T)
    # compute the translation matrix
    tra = np.eye(4)
    tra[:3, 3] = -origin#.ravel()
    tra[3, 3] = 1

    # combine both to full homogeneous transformation matrix
    transform = np.dot(rot, tra)

    return transform


def apply_transform(M, pos):
    # convert to homogeneous coordinates
    num, dim = pos.shape
    hom = np.ones((num,dim+1))
    hom[:,:3] = pos
    # apply transformation
    hom = (M.dot(hom.T)).T
    # backconversion
    pos = np.array([hom[i,:3] / hom[i,3] for i in range(hom.shape[0])])
    return pos

def apply(transformation, old):
    m, n = old.shape
    old = np.hstack((old, np.ones((m, 1))))
    new = np.dot(old, transformation.T)
    new = new[:, :3]
    return new
