from config.imports import * 

"""
==================================================================================================================
Problem Configuration
==================================================================================================================
"""

inputfolder  = "./workfolder/Modes/"
outputfolder = "./workfolder/Modes/"

os.makedirs(inputfolder, exist_ok=True)
os.makedirs(outputfolder, exist_ok=True)

### Beam
mesh = BoxMesh(Point(0., 0., 0.), Point(1., 0.1, 0.04), 60, 10, 5)

### Sub domain for clamp at left end
def DirichletBoundary(x, on_boundary):
    return near(x[0], 0.) and on_boundary

### Sub domain for excitation at right end
def NeumannBoundary(x, on_boundary):
    return near(x[0], 1.) and on_boundary

config = {
    'verbose'           :   True,
    'inputfolder'       :   inputfolder,
    'outputfolder'      :   outputfolder,
    'export_vtk'        :   False,

    'FinalTime'         :   5,
    'nTimeSteps'        :   250,

    'mesh'              :   mesh,
    'DirichletBoundary' :   DirichletBoundary,
    'NeumannBoundary'   :   NeumannBoundary,

    'infmode'           :   True,
    'zener_kernel'      :   True,

    ### Material parameters
    'Young'             :   1.e3,
    'Poisson'           :   0.3,
    'density'           :   1.,

    ### Viscous term
    'viscosity'         :   True,
    'two_kernels'       :   False,

    ### Measurements
    'observer'          :   TipYDisplacementObserver,
}