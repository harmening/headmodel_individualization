#!/usr/bin/env python
import numpy as np


def load_elecs_txt(fn):
    with open(fn, 'r') as f:
        elecs = [[float(i) for i in line.split()] for line in f.readlines()]
    return np.array(elecs)



def load_tri(filename, print_warnings=False):
    """ Loads mesh from tri-file
    Parameters
    ----------
    filename: str
        Path to surface ASCII file (ending with '.tri').
    Returns
    -------
    pos : array, shape=(n_vertices, 3)
        Coordinate points.
    tri : int array, shape=(n_faces, 3)
        Triangulation (each line contains indices for three points which
        together form a face).
    """
    with open(filename, "r") as fid:
        lines = fid.readlines()
    n_nodes = int(lines[0].split()[-1])
    n_tris = int(lines[n_nodes + 1].split()[-1])
    n_items = len(lines[1].split())
    if n_items in [3, 6, 14, 17]:
        inds = range(3)
    elif n_items in [4, 7]:
        inds = range(1, 4)
    else:
        raise IOError('Unrecognized format of data.')
    pos = np.array([np.array([float(v) for v in l.split()])[inds]
                   for l in lines[1:n_nodes + 1]])
    tris = np.array([[int(l.split()[ind]) for ind in inds]
                     for l in lines[n_nodes + 2:n_nodes + 2 + n_tris]])
    tris -= 1
    if not n_items in [3, 4] and print_warnings:
        print('Node normals were not read.')
    # ensure that tris start at zero
    if np.min(tris) != 0:
        tris -= np.min(tris)
    return pos, tris


def write_tri(pos, tri, filename, normals=None):
    """ Write mesh into tri-file
    Parameters
    ----------
    pos : array, shape=(n_vertices, 3)
        Coordinate points.
    tri : int array, shape=(n_faces, 3)
        Triangulation (each line contains indices for three points which
        together form a face).
    filename : str
        Path for storing surface file (ending with '.tri').
    """
    min_idx = np.min(np.array(tri).flatten())
    pos = np.array(pos)
    tri = np.array(tri, dtype=int)
    if isinstance(normals, list) or isinstance(normals, np.ndarray):
        norm = normals
    else:
        norm = get_normals(pos, tri)    
    with open(filename, 'w') as f:
        f.write('- '+str(pos.shape[0])+'\n')
        for ii in range(pos.shape[0]):
            pnts = ' '.join([str(pos[ii][i]) for i in range(3)])
            norms = ' '.join([str(norm[ii][i]) for i in range(3)])
            f.write(pnts+' '+norms+'\n')
        f.write('-'+(' '+str(tri.shape[0]))*3+'\n')
        for ii in range(tri.shape[0]):
            f.write(' '.join([str(tri[ii][i]-min_idx) for i in range(3)])+'\n')
    return



def calc_normal(p1, p2, p3):
    """ Calculate the surface normal of triangle
    Parameters
    ----------
    p1, p2, p3, : three np.arrays (shape: all 1x3 or 3x1)
        Point coordinates of triangle
    Parameters
    ----------
    n : np.array, shape 1x3
        Normal vector
    """
    return np.cross(p2-p1, p3-p1)

def normals_for_faces(vertices, faces):
    normals_f = np.zeros(faces.shape)
    # ensure that tris start at zero
    if np.min(faces) != 0:
        tri_min = np.min(faces)
        faces -= tri_min
    for i, tri in enumerate(faces):
        pos = [vertices[tri[ii]] for ii in range(3)]
        normals_f[i,:] = calc_normal(*pos)
    return normals_f

def get_normals(vertices, faces):
    normals_v = normals_for_faces(vertices, faces)
    if normals_v.shape == faces.shape:
        normals_v = vertex_normals(faces, normals_v)

    normals_v = verts_normals_orientation(vertices, faces,
                                          normals_v, normalsIn=True)
    ### NEW: CHECK FOR NANS (+ dirty fix)
    nan_idx = set(np.argwhere(np.isnan(normals_v))[:,0])
    for idx in nan_idx:
        neighbors = set([faces[t][i] for t in np.argwhere(faces==idx)[:,0] for i in range(3)])  - {idx}
        nrm = np.array([0.0, 0.0, 0.0])
        for n in neighbors:
            if (not np.isnan(normals_v[n]).any()):
                nrm += normals_v[n]
        nrm /= np.linalg.norm(nrm)
        normals_v[idx] = nrm
    assert (not np.isnan(normals_v).all())
    #assert (np.linalg.norm(normals_v, axis=1)==1.0).all()
    ### END NEW
    return normals_v

def surface_area(vertices, faces):
    x1= vertices[faces[:,0],0]
    y1= vertices[faces[:,0],1]
    z1= vertices[faces[:,0],2]
    x2= vertices[faces[:,1],0]
    y2= vertices[faces[:,1],1]
    z2= vertices[faces[:,1],2]
    x3= vertices[faces[:,2],0]
    y3= vertices[faces[:,2],1]
    z3= vertices[faces[:,2],2]
    area = np.sqrt(pow((y2-y1)*(z3-z1)-(y3-y1)*(z2-z1), 2) +
		   pow((z2-z1)*(x3-x1)-(z3-z1)*(x2-x1), 2) +
		   pow((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1), 2))
    return sum(area)

def verts_normals_orientation(vertices, faces, normals, normalsIn):
    area1 = surface_area(vertices,faces)
    area2 = surface_area(vertices+normals,faces)
    if area2 < area1:
        faces = faces[:,::-1]
        #normals = surface_normals(vertices,faces)
        normals_f = normals_for_faces(vertices, faces)
        normals = vertex_normals(faces, normals_f)
    if normalsIn:
        faces = faces[:,::-1]
        #normals = surface_normals(vertices,faces)
        normals_f = normals_for_faces(vertices, faces)
        normals = vertex_normals(faces, normals_f)
    for i in range(normals.shape[0]):
        norm_i = np.sqrt(np.sum([pow(normals[i,ii], 2) for ii in range(3)]))
        normals[i,:] /= norm_i
    return normals


def vertex_normals(faces, face_normals):
    normals = np.zeros((faces.max()+1, 3))
    """
    summed = np.zeros((vertex_count, 3))
    for face, normal in zip(faces, face_normals):
        summed[face] += normal

    """
    for i in range(faces.max()+1):
        tris = np.argwhere(faces == i)[:,0]
        normals[i,:] += np.sum(face_normals[np.ix_(tris)], axis=0)
        norm_i = np.sqrt(np.sum([pow(normals[i,ii], 2) for ii in range(3)]))
        normals[i,:] /= norm_i
    return normals
