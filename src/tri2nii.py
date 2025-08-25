#!/usr/bin/env python
"""
Converts .tri files to .nifti segmentation masks
"""
import os
import nibabel as nib
import nilearn, nilearn.image
import numpy as np
import argparse
from tqdm import tqdm

from src.tri_io import load_tri
import tempfile
import trimesh
import vtk
import scipy.io as sio



def is_inside(points, polydata):
    """
    This function is provided a point in space, and a triangular mesh provided
    via an stl file, and than uses the vtkSelectEnclosedPoints module to return
    whether or not the points lies within the mesh.

        Parameters:
            points: Points to decide on whether is lies within the mesh or not
            polydata: vtk polydata instance of the triangular surface mesh

        Returns:
            boolean: is the point within the mesh
    """
    vtkPoints = vtk.vtkPoints()
    for point in points:
        vtkPoints.InsertNextPoint(point)

    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(vtkPoints)


    selectEnclosedPoints = vtk.vtkSelectEnclosedPoints()
    selectEnclosedPoints.SetInputData(pointsPolydata)
    selectEnclosedPoints.SetSurfaceData(polydata)
    selectEnclosedPoints.Update()

    return [selectEnclosedPoints.IsInside(i) == 1 for i in range(len(points))]


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


def bnd2stl_fn(bnd, stl_fn):
    """ Writes surface mesh to stl file
    """
    pos, tris = bnd
    pos, tris = np.array(pos), np.array(tris)
    mesh = trimesh.Trimesh(pos, tris)
    normals = np.zeros(tris.shape)
    for t, tri in enumerate(tris):
        p1, p2, p3 = [np.array(pos[tri[i],:]) for i in range(3)]
        normals[t,:] = calc_normal(p1, p2, p3)

    mesh.__init__(vertices=bnd[0], faces=bnd[1], face_normals=normals)
    mesh.export(file_obj=stl_fn, file_type='stl_ascii')
    return


def tri2nii(bnds, output_dir=None, transform=np.eye(4), t1_fn='template.nii', meshes='all'):

    # Load T1 image
    t1 = nib.load(t1_fn)
    assert all([t1.affine[i,i] == 1.0 for i in range(4)])

    # load meshes

    # Prepare segmentation mask
    data = np.zeros(t1.shape)

    # Apply padding
    pad_width = ((0, 0),  # x-axis padding
                (0, 0),  # y-axis padding
                (0, 10))  # z-axis padding (on the positive side)

    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)

    # Transform points being inside surface meshes into index space for each tissue type
    # Go from out to inside - this makes sure that every voxel has only one label.
    # {1: 'scalp', 2: 'skull', 3: 'csf', 4: 'cortex'}
    for tissue_label in range(1, len(bnds)+1):
        tissue_coords = bnds[tissue_label-1][0]
        ## Transform to voxel space
        # Note that it seems that SimNibs msh2nii just uses t1.shape / 2 as offset
        # bias instead of the actual t1.affine values

        # Note that one should apply the inverse t1.affine to the tissue_coords.
        # This is in our application not necessary, as the t1.affine is for our
        # data just a translation (because our inputs are all in the ACPC
        # coordinate system).

        # Apply translation of t1.affine (half of the shape)
        nx, ny, nz = t1.shape
        assert all([-int(ni/2) for ni in t1.shape] == t1.affine[:3,-1])
        tissue_coords[:,0] += int(nx / 2.)
        tissue_coords[:,1] += int(ny / 2.)
        tissue_coords[:,2] += int(nz / 2.)

        # Get complete range of XYZ coords for all nodes for later speed up
        x_min = int(np.min(tissue_coords[:,0]))
        x_max = min(int(np.max(tissue_coords[:,0])), nx)
        y_min = int(np.min(tissue_coords[:,1]))
        y_max = min(int(np.max(tissue_coords[:,1])), ny)
        z_min = int(np.min(tissue_coords[:,2]))
        z_max = min(int(np.max(tissue_coords[:,2])), nz)

        #print('min/max X coordinate: ', x_min,x_max)
        #print('min/max Y coordinate: ', y_min,y_max)
        #print('min/max Z coordinate: ', z_min,z_max)


        # Pre sorting to not check every voxel
        # Fixed voxel size to 1mm here!
        points = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    p = np.array([i,j,k])
                    if (p[0] >= x_min and p[0] <= x_max and
                        p[1] >= y_min and p[1] <= y_max and
                        p[2] >= z_min and p[2] <= z_max):
                        points.append(p)


        # Create vtk polydata instance of surface mesh
        dirpath = tempfile.mkdtemp()
        stl_fn = os.path.join(dirpath, "stl_file.stl")
        bnd2stl_fn(bnds[tissue_label-1], stl_fn)
        readerSTL = vtk.vtkSTLReader()
        readerSTL.SetFileName(stl_fn)
        readerSTL.Update()

        polydata = readerSTL.GetOutput()

        # Check if pre sorted points are inside the mesh
        inside = is_inside(points, polydata)

        # Fill the data array with the tissue label
        count = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    p = np.array([i,j,k])
                    if (p[0] >= x_min and p[0] <= x_max and
                        p[1] >= y_min and p[1] <= y_max and
                        p[2] >= z_min and p[2] <= z_max):
                        if inside[count]:
                            data[i][j][k] = tissue_label
                        count += 1


    ## Save the segmentation mask (as one int-mask or as one binary mask per tissue label)
    affine = transform @ t1.affine

    if meshes == 'all':
        # {1: 'scalp', 2: 'skull', 3: 'csf', 4: 'cortex'}
        for tissue_label in range(1, len(bnds)+1):
            new_data = np.zeros(data.shape)
            new_data[data == tissue_label] = 1

            #new_img = nilearn.image.new_img_like(t1,new_data,affine=affine)
            new_img = nib.nifti1.Nifti1Image(new_data, affine, t1.header)
            new_img.header['dim'][1] += sum(pad_width[0])
            new_img.header['dim'][2] += sum(pad_width[1])
            new_img.header['dim'][3] += sum(pad_width[2])
            if output_dir != None:
                output_file = output_dir + '/mask_'+str(tissue_label)+'.nii.gz'
            else:
                output_file = '_mask_'+str(tissue_label)+'.nii.gz'
            #print('Output File:', output_file)
            nib.save(new_img, output_file)
    else:
        tissue_color = {1: 0.45, # ~0.4–0.6 (depends on fat content)
                        2: 0.05, # ~0.0–0.1 (very dark, almost no signal)
                        3: 0.1, #CSF ~0.0–0.1 (dark; long T1 → low signal)
                        4: 0.55} #cortex (0.5–0.7 (brighter than cortex)
        new_data = np.zeros(data.shape)
        for tissue_label in range(1, len(bnds)+1):
            new_data[data == tissue_label] = tissue_color[tissue_label]
        #new_img = nilearn.image.new_img_like(t1,new_data,affine=affine)
        new_img = nib.nifti1.Nifti1Image(new_data, affine, t1.header)
        new_img.header['dim'][1] += sum(pad_width[0])
        new_img.header['dim'][2] += sum(pad_width[1])
        new_img.header['dim'][3] += sum(pad_width[2])
        if output_dir != None:
            output_file = output_dir + '/T1.mgz'
        else:
            output_file = 'bnd_w_mask.nii.gz'
        nib.save(new_img, output_file)



