"""Module allowing for ray-casting operations and measurements

This module contains the `rayCaster` class which allows for ray-casting
operations to be performed on any mesh represented by a 'vtkPolyData'
object.

The class uses the `vtkOBBTree` class to perform ray-casting
and can calculate the coordinates of the entry/exit points between the
ray and the surface. It can in addition calculate the distance a ray
travels within the closed section of the surface, i.e., within the solid

The class features a static method 'fromSTL' which allows for it to be
initialized directly from an STL file which it loads and extracts the polydata
from initializing the ray-caster.
"""

import vtk
import logging as log
import math

class rayCaster(object):
    r"""Ray-casting class

    The class uses the `vtkOBBTree` class to perform ray-casting
    and can calculate the coordinates of the entry/exit points between the
    ray and the surface. It can in addition calculate the distance a ray
    travels within the closed section of the surface, i.e., within the solid

    Note:
        The class features a static method 'fromSTL' which allows for it to be
        initialized directly from an STL file which it loads and extracts the
        polydata from initializing the ray-caster.

    Attributes:
        mesh (vtkPolyData): The surface mesh provided either directly when
            initializing the class or loaded from an STL file through the
            'fromSTL' static-method.
        caster (vtkOBBTree): The `vtkOBBTree` object which performs the
            intersection operations between lines and the surface stored under
            'mesh'
    """

    def __init__(self, mesh):
        r"""Class constructor and initialization

        This method creates a new 'rayCaster' object with a given 'mesh' which
        is a 'vtkPolyData' object containing the mesh ray-casting is performed
        on

        Parameters:
            mesh (vtkPolyData): The mesh ray-casting is performed on

        Returns:
            A new 'rayCaster' object
        """

        self.mesh = mesh  # set the 'mesh'
        self.caster = None
        self._initCaster()  # create a caster

    @classmethod
    def fromSTL(cls, filenameSTL, scale=1.0):
        r"""Create a 'rayCaster' object from an .stl file

        This static method allows for the creation of a 'rayCaster' object
        directly from an .stl file under 'filenameSTL'. The method loads the STL
        data and (optionally) scales them, i.e., changes the measurement units
        before returning a 'rayCaster' object

        Parameters:
            filenameSTL (str): The full path to the .stl file
            scale (float): The scale that will be applied to the polydata, i.e.,
                multiplied by 'scale'. If this is set to "1.0" then no scaling
                is applied. This parameter can	be used to change the
                measurement units of the STL mesh, e.g. if the original mesh is
                in mm then a scale of "1.0e-3" will change it to meters

        Returns:
            A 'rayCaster' object with 'mesh' being set to the 'vtkPolyData'
            loaded from the 'filenameSTL' file and (optionally) multiplied by
            'scale'
        """

        readerSTL = vtk.vtkSTLReader()  # create a 'vtkSTLReader' object
        readerSTL.SetFileName(
            filenameSTL)  # set the .stl filename in the reader
        readerSTL.Update()  # 'update' the reader i.e. read the .stl file

        polydata = readerSTL.GetOutput()

        # If there are no points in 'vtkPolyData' something went wrong
        if polydata.GetNumberOfPoints() == 0:
            raise ValueError(
                "No point data could be loaded from '" + filenameSTL)
            return None

        # Create a new 'rayCaster' with the mesh loaded from the .stl file
        rT = cls(polydata)

        #if a non-unit 'scale' is given then scale the polydata
        if scale != 1.0:
            rT.scaleMesh(scale)

        #return the 'rayCaster' object
        return rT

    def scaleMesh(self, scale):
        r"""Scales (multiplies) the 'mesh' by 'scale'

        This method scales, i.e, multiplies the 'mesh' vtkPolyData of the object
        by 'scale'

        Parameters:
            scale (float): The scale that will be applied to the polydata.
                This parameter can be used to change the measurement units of
                the STL mesh, e.g. if the original mesh is in mm then a scale of
                "1.0e-3" will change it to meters
        """

        #create a 'vtkTransform' object
        transform = vtk.vtkTransform()
        #create a 'vtkTransformPolyDataFilter' object
        transformFilter = vtk.vtkTransformPolyDataFilter()

        #set the scale transformation
        transform.Scale(scale, scale, scale)  #assuming uniform scale

        #set the 'mesh' as the input of the 'vtkTransformPolyDataFilter'
        transformFilter.SetInput(self.mesh)
        #set 'transform' as the transformation of the 'vtkTransformPolyDataFilter'
        transformFilter.SetTransform(transform)
        #update to apply the transformation
        transformFilter.Update()

        #replace the existing 'mesh' with the scaled one
        self.mesh = transformFilter.GetOutput()
        #update the caster since the 'mesh' changed
        self._updateCaster()

    def _initCaster(self):
        r"""Create a 'caster'

        This is an internal method. DO NOT USE DIRECTLY. This method creates a
        'caster', i.e., a 'vtkOBBTree' object which is used later on to cast
        rays.
        """

        #create a 'vtkOBBTree' object
        self.caster = vtk.vtkOBBTree()
        #set the 'mesh' as the caster's dataset
        self.caster.SetDataSet(self.mesh)
        #build a caster locator
        self.caster.BuildLocator()

    def _updateCaster(self):
        r"""Update the 'caster'

        This is an internal method. DO NOT USE DIRECTLY. This method updates the
        'caster', and it is called when the 'mesh' has been changed, e.g. after
        scaling
        """

        self.caster.SetDataSet(self.mesh)
        self.caster.BuildLocator()

    def castRay(self, pointRaySource, pointRayTarget):
        r"""Perform ray-casting for a given ray

        This method performs ray-casting for a ray emanating from a point
        'pointRaySource' towards a point 'pointRayTarget'. It then returns a
        list of lists where each one represents a point where this ray
        intersects with the 'mesh' this 'rayCaster' object was created with.

        Parameters:
            pointRaySource (list, or tuple): Contains the 3 coordinates of the
                point the ray is cast from
            pointRayTarget (list, or tuple): Contains the 3 coordinates of the
                point the ray is cast towards

        Returns:
            A list of list objects, each of which contains the coordinates of an
            intersection point between the ray and the 'mesh'. If the returned
            list is empty then no intersection points were found for the given
            configuration
        """

        #create a 'vtkPoints' object to store the intersection points
        pointsVTKintersection = vtk.vtkPoints()

        #perform ray-casting (intersect a line with the mesh)
        code = self.caster.IntersectWithLine(pointRaySource,
                                             pointRayTarget,
                                             pointsVTKintersection, None)

        # Interpret the 'code'. If "0" is returned then no intersections points
        # were found so return an empty list
        if code == 0:
            log.info(
                "No intersection points found for 'pointRaySource': " + str(
                    pointRaySource) + " and 'pointRayTarget': " + str(
                    pointRayTarget))
            return []
        # If code == -1 then 'pointRaySource' lies outside the surface
        elif code == -1:
            log.info("The point 'pointRaySource': " + str(
                pointRaySource) + "lies inside the surface")


        #get the actual data of the intersection points (the point tuples)
        pointsVTKIntersectionData = pointsVTKintersection.GetData()
        #get the number of tuples
        noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()

        #create an empty list that will contain all list objects
        pointsIntersection = []

        # Convert the intersection points to a list of list objects.
        for idx in range(noPointsVTKIntersection):
            _tup = pointsVTKIntersectionData.GetTuple3(idx)
            pointsIntersection.append(_tup)

        #return the list of list objects
        return pointsIntersection

    def isPointInside(self, point):
        r"""Checks if a point lies inside the 'mesh' surface or not

        This method checks whether a 'point' lies inside a surface or not and
        returns "True" if does and "False" otherwise

        Parameters:
            point (list, or tuple): Contains the 3 coordinates of the point to
                be evaluated

        Returns:
            "True" if the point lies inside the surface and "False" if the point
            lies either outside the surface or its location can not be 'decided'
            , e.g., the point is exactly on a vertex of the surface
        """

        code = self.caster.InsideOrOutside(point)

        if code == -1:  #point is inside
            return True
        else:  #point is either outside the surface or can not be located
            return False

    def _calcDistance(self, pointA, pointB):
        r"""Calculate the euclidean distance between two points"""

        distance = math.sqrt((pointA[0] - pointB[0]) ** 2 +
                             (pointA[1] - pointB[1]) ** 2 +
                             (pointA[2] - pointB[2]) ** 2)

        return distance

    def calcDistanceInSolid(self, pointRaySource, pointRayTarget):
        r"""Calculates the distance a ray travels within a solid

        This method uses the 'castRay' method and performs ray-casting for the
        given points and calculates and returns the distance a given ray travels
        inside a closed surface, i.e., a solid. The method accounts for all
        possible relative locations of the source and target points of the ray
        (in regards to the surface through the 'isPointInside' method) so these
        can either be inside or outside the surface

        Parameters:
            pointRaySource (list, or tuple): Contains the 3 coordinates of the
                point the ray is cast from
            pointRayTarget (list, or tuple): Contains the 3 coordinates of the
                point the ray is cast towards

        Returns:
            A number (float) with the total distance the ray traveled through
            the solid
        """

        # Perform ray-casting with the given ray source and target
        pointsIntersection = self.castRay(pointRaySource, pointRayTarget)
        # Get the number of intersection points
        noPoints = len(pointsIntersection)

        # Check whether the source or target points lie within the surface/solid
        isSourceIn = self.isPointInside(pointRaySource)
        isTargetIn = self.isPointInside(pointRayTarget)

        # If no intersection points were found for this ray then either both the
        # source and target are in and the traveled distance is the distance
        # between source and target or both points are outside and the traveled
        # distance within the solid is "0.0"
        if noPoints == 0:
            if isSourceIn and isTargetIn:
                return self._calcDistance(pointRaySource, pointRayTarget)
            else:
                return 0.0

        distance = 0.0  #initialize distance to 0

        # If an odd number of intersection points was found for this ray then
        # either the source is in and the first part of the traveled distance is
        # equal to the distance between the source and the first intersection
        # point (exit point) or the target is in and the last part of the
        # traveled distance is equal to the distance between the target and the
        # last intersection point (entry point)
        if noPoints % 2 == 1:
            if isSourceIn:
                distance = self._calcDistance(pointRaySource,
                                              pointsIntersection[0])
            elif isTargetIn:
                distance = self._calcDistance(pointRayTarget,
                                              pointsIntersection[-1])

            # If only one intersection point was found then the distance
            # calculated above is the total distance and it's returned otherwise
            # there's at least one more pair of intersection points to be
            # accounted for.
            if noPoints == 1:
                return distance
            else:
                # If the source is in then start looping from 2nd point till the
                # last one. Otherwise if the target is in then start looping
                # from 1st point till the last one but in either case the looped
                # pairs can be added in an entry-exit pattern
                if isSourceIn:
                    loopStart = 1
                    loopStop = noPoints
                elif isTargetIn:
                    loopStart = 0
                    loopStop = noPoints - 1
                # Loop in pairs
                for idxPoint in range(loopStart, loopStop, 2):
                    distance += self._calcDistance(
                        pointsIntersection[idxPoint],
                        pointsIntersection[idxPoint + 1])

        # If an even number of intersection points was found for this ray
        # (the possibility of 0 points was accounted for above) then either both
        # the source and target are in and the first and last point are
        # accounted for in a manner similar to the case of 'noPoints == 0' or
        # both points are outside and every point-pair in between should follow
        # an entry-exit pattern
        if noPoints % 2 == 0:
            if isSourceIn and isTargetIn:
                distance += self._calcDistance(pointRaySource,
                                               pointsIntersection[0])
                distance += self._calcDistance(pointRayTarget,
                                               pointsIntersection[-1])
                loopStart = 1
                loopStop = noPoints - 1
            else:
                loopStart = 0
                loopStop = noPoints
            # Loop in pairs
            for idxPoint in range(loopStart, loopStop, 2):
                distance += self._calcDistance(
                    pointsIntersection[idxPoint],
                    pointsIntersection[
                        idxPoint + 1])

        return distance

