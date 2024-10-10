# Headmodel Individualization
**Data-driven head model individualization from exact electrode positions or photogrammetry of the subject's head surface improves M/EEG source localization accuracy!**<br>
<!--- 
Supplementary code to the scientific publication ["Head model individuzalization"]().**<br>
--->


## Individualization algorithm
`pca_surfacemesh_warping.py` contains the main function of how to start the inidividualization. It is based on a low-dimensional representation (PCA) of head shape surface meshes trained on a equally segmented and triangulated MRI database of 316 subjects. Warping is done by finding weights for the PCs by minimizing the shape difference between electrodes / scalp proxies and fitted scalp. 
It contains the following steps:
* Upload your fiducials (NSA, LPA, RPA) and scalp proxy, i.e. sensor positions or any points of the scalp. Ideally points above the ears.
* Using the fiducials, transform the points to the [CTF-coordinate system](https://www.fieldtriptoolbox.org/faq/coordsys/), since the database on which the PCA was applied, lives in CTF space.
* PCA warping, parameters are: number of PCs used for reconstruction, regularizer type (if meshes are intersecting)
* Transform back from CTF in original input space.
* Save surface meshes as .tri (for [OpenMEEG](https://openmeeg.github.io/)) and as .mat for [Matlab](https://de.mathworks.com/products/matlab.html)/[FieldTrip](https://www.fieldtriptoolbox.org/) or any other mesh format like .stl, .obj, .ply, ....






<!--- 
## Citing
If you find the headmodel individualization useful for your research, please consider citing our related [paper]().
```
@article{Harmening_2024,
      author  = {Harmening, Nils and
                 von Lühmann, Alexander and
                 Blankertz, Benjamin}
      title   = {Data-driven head model individualization from exact electrode positions or photogrammetry improves M/EEG source localization accuracy}
      year    = {2024},
      journal = {Nice Journal}
      doi     = {},
      volume  = {},
      number  = {},
      pages   = {},
}
```
--->
