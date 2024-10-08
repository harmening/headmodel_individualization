import numpy as np

def transform_to_ctf(pts, nas, lpa, rpa, return_transform=False):
    transform = get_transform(nas, lpa, rpa)
    pts_ctf = apply_transform(transform, pts)
    if return_transform:
        return pts_ctf, transform
    return pts_ctf
    

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
