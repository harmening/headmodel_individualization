# bcas PCA database (Chinese adults)

Head-shape basis built from 178 healthy adults of Chinese ancestry, aged 22 to
79, 77 male and 103 female. Acquired at the Clinical Imaging Research Centre,
National University of Singapore, on a 3T Siemens Magnetom Trio Tim.

The source cohort has 180 subjects. Two were excluded after QC (`sub-0176` and
`sub-0007`), both looking like the nonlinear fiducial warp of the automatic MRI
segmentation pipeline failed.

`../bcas_hartmut/` is the same subjects with the neck-extended HArtMuT scalp.

Source: https://www.nitrc.org/projects/adultatlas (NITRC project `adultatlas`)

"bcas" is the prefix of the download archives on that page, used here as a short
folder name. The authors do not define an abbreviation for the dataset, so cite
the full title below rather than "bcas".

```
Zhu, J. and Qiu, A. Chinese adult brain atlas with functional and white
matter parcellation. Scientific Data 9, 352 (2022).
https://doi.org/10.1038/s41597-022-01476-2
```


## Attribution, required

The source data is released under a Creative Commons Attribution licence. **This
credit has to travel with any redistribution of this basis or anything derived
from it:**

> Chinese Adult Brain Atlas subject data, by Jingwen Zhu and Anqi Qiu, National
> University of Singapore, CC BY, https://www.nitrc.org/projects/adultatlas

CC BY also asks that changes be indicated. No source image data is redistributed
here. The T1 volumes were resliced to standard RAS, segmented with SPM12 against
the extended tissue probability maps of Huang et al. (2013), ACPC-reoriented,
given fiducials by nonlinear warping of template landmarks, converted to boundary
surfaces by star projection onto a fixed sphere tessellation, transformed into CTF
head coordinates, and reduced to the principal components stored here. Pipeline:
https://github.com/harmening/MRIsegmentation
