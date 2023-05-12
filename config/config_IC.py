from config.imports import * 

"""
==================================================================================================================
Problem Configuration
==================================================================================================================
"""

inputfolder  = "./workfolder/IC/"
outputfolder = "./workfolder/IC/"

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

cutoff_time = 1.
magnitude   = 1.
tmax        = 4/5
tzero       = 1.
load_Bending = Expression(("0", "t <= tm ? p0*t/tm : (t <= tz ? p0*(1 - (t-tm)/(tz-tm)) : 0)", "0"), t=0, tm=tmax, tz=tzero, p0=magnitude, degree=0) ### Bending

config = {
    'verbose'           :   True,
    'inputfolder'       :   inputfolder,
    'outputfolder'      :   outputfolder,
    'export_vtk'        :   False,

    'FinalTime'         :   50,
    'nTimeSteps'        :   10, ### Steps per time unit!!!

    'mesh'              :   mesh,
    'DirichletBoundary' :   DirichletBoundary,
    'NeumannBoundary'   :   NeumannBoundary,
    'loading'           :   load_Bending, ###  load_Bending, [load_Bending, load_Extension]

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
    'observer'          :   TipDisplacementObserver,
}