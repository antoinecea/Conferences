# CFM 2025, presentation

This repository is a companion to the CFM 2025 conference presentation
Here are the files relative to the simulation performed in the presentation

## Directory 'mfront_laws'

MFront files of the laws, to compile : 'mfront --obuild --interface=generic file.mfront' in the directory 'mfront_laws'.

### Linear case

 - MoriTanaka.mfront : Mori Tanaka scheme, oriented distribution (used with the FEniCSx simulation "linear_hole_plate.py"
 
### Linear viscoelastic cases

 - Idiart_elastic_inclusions.mfront : Idiart scheme in linear visco-elasticity, spherical elastic inclusions (used with the FEniCSx simulation "non_linear_hole_plate.py")
 
 - Idiart_random_elastic_inclusions.mfront : Idiart scheme in linear visco-elasticity, randomly oriented elastic fibers (used with the FEniCSx simulation "non_linear_hole_plate.py"). The number of orientations used to represent the microstructure can be changed by setting '@IntegerConstant Nr=..;' in the mfront file. All the orientations of the fibers are available in the file 'extra-headers/TFEL/Material/microstructure.hxx'

### Non-linear case

 - Molinari_Explicit.mfront : Additive law used by Mercier et al (2019), explicit integration in time. Non-linear viscoplastic matrix and spherical elastic inclusions. The matrix law used in the presentation is in the file 'Perzyna.mfront'. The 'Molinari_Explicit' behaviour is used in the FEniCSx simulation "non_linear_hole_plate.py".
 
## Current directory

 - "plate_hole_20k.xdmf": mesh of the holed plate with 20k elements
 - "plate_hole_40k.xdmf": mesh of the holed plate with 40k elements
 - "linear_hole_plate.py": FEniCSx simulation using a linear solver
 - "non_linear_hole_plate.py": FEniCSx simulation using a non linear SNES solver

# Authors and acknowledgment

Antoine Martin, Thomas Helfer

The authors acknowledge the ANR Agency for the financial support of the AnoHonA project (n° AAPG2023),
within the framework of which this research was conducted.

