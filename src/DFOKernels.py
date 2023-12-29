import numpy as np
import torch
from torch import nn
from math import gamma
import matplotlib.pyplot as plt 

from RationalApproximation import RationalApproximation_AAA
"""
==================================================================================================================
MultiTermKernel for a interval 0 to 1 or 1 to 2
==================================================================================================================
"""

class MultiTermKernel: 

    def __init__(self, phis, alphas, **kwargs):
        self.phis = phis
        self.alphas = alphas

        self.RATarget = np.vectorize(self.eval_RATarget) 
        self.kernel = np.vectorize(self.eval_kernel)

    def eval_kernel(self, t):
        return np.sum(self.phis / gamma(1 - self.alphas) * t **(-self.alphas))
    
    def eval_RATarget(self, t):
        return np.sum(self.phis * t ** (1 - self.alphas))
    

"""
==================================================================================================================
Class to handle distributed fractional order kernels 
==================================================================================================================
"""

class DFOKernel(nn.Module): 
    def __init__(self, alphas01, alphas12, phis01, phis12, phi0, phi1, phi2, **kwargs): 
        super().__init__()

        self.alphas01 = alphas01
        self.alphas12 = alphas12

        self.phis01 = phis01
        self.phis12 = phis12

        self.phi0 = phi0
        self.phi1 = phi1
        self.phi2 = phi2

        self.MTK01 = MultiTermKernel(alphas01, phis01)
        self.MTK12 = MultiTermKernel(alphas12, phis12)

        self.calculate_rational_approximations(**kwargs)

    def calculate_rational_approximations(self, **kwargs): 
        t_final = kwargs.get("t_final", 1)
        dt = kwargs.get("dt", 1e-6)
        Zmin, Zmax = 1/t_final, 1/dt 
        tol = kwargs.get("tol", 1e-10)
        maxDegree = kwargs.get("max_degree", 30)
        nSupportPoints = kwargs.get("nSupportPoints", 100)

        self.RA01 = RationalApproximation_AAA(alpha=0.5, tol=tol, MaxDegree=maxDegree, nSupportPoints=nSupportPoints, Zmin= Zmin, Zmax= Zmax, verbose=False, 
                                    TargetFunction=self.MTK01.RATarget)
        self.RA12 = RationalApproximation_AAA(alpha=0.5, tol=tol, MaxDegree=maxDegree, nSupportPoints=nSupportPoints, Zmin= Zmin, Zmax= Zmax, verbose=False, 
                                    TargetFunction=self.MTK12.RATarget)

    
    @classmethod
    def create_from_distribution(cls, *, distribution, **kwargs):
        
        # sort support and check for 0, 1 and 2 
        support = np.asarray(kwargs.get("support", (0,2)))
        support.sort()
        count_012 = 1 * (0 == support[0]) + 1 * (1 >= support[0] and 1 <= support[1]) + 1 * (2 == support[1])         
        
        # get quadrature weight 
        n_QP = np.asarray(kwargs.get("n_QP", 100))
        qweight = 2 / (n_QP + count_012)

        # get weights values at 0, 1 and 2
        phi0 = qweight * distribution(0) if 0 == support[0] else np.zeros(1) 
        phi1 = qweight * distribution(1) if 1 >= support[0] and 1 <= support[1] else np.zeros(1) 
        phi2 = qweight * distribution(2) if 2 == support[1] else np.zeros(1) 

        # get quadrature points 
        qpoints01 = np.linspace(0.5 / n_QP, 1 - 0.5/n_QP, n_QP)
        qpoints12 = np.linspace(1 + 0.5/n_QP, 2 - 0.5/n_QP, n_QP)
        
        # get weighted values at quadrature points, 0 if outside of support 
        phis01 = qweight * distribution(qpoints01) * (support[0] < qpoints01) * (qpoints01 < support[1])
        phis12 = qweight * distribution(qpoints12) * (support[0] < qpoints12) * (qpoints12 < support[1])
        
        # eliminate zero entries 
        qpoints01 = qpoints01[phis01 != 0]
        phis01 = phis01[phis01 != 0]
        qpoints12 = qpoints12[phis12 != 0]
        phis12 = phis12[phis12 != 0]

        # call constructor 
        return cls(qpoints01, qpoints12, phis01, phis12, phi0, phi1, phi2)
    
   

    # TODO 
    @classmethod
    def create_form_quadrature_rule(cls, qpoints, qvalues): 
        
        # quadrature weight = interval length / number of points  
        qweight = (support[1] - support[0]) / qpoints.shape[0]
        
        # get quadrature points and values 
        qpoints01 = qpoints[(0 < qpoints) * (qpoints < 1)]
        qpoints12 = qpoints[(1 < qpoints) * (qpoints < 2)]

        phis01 = qvalues[(0 < qpoints) * (qpoints < 1)] * qweight
        phis12 = qvalues[(1 < qpoints) * (qpoints < 2)] * qweight

        phi0 = qvalues[qpoints==0] * qweight if 0 in qpoints else np.zeros(1) 
        phi1 = qvalues[qpoints==1] * qweight if 1 in qpoints else np.zeros(1)
        phi2 = qvalues[qpoints==2] * qweight if 2 in qpoints else np.zeros(1)

        # get support from quadrature points 
        support = np.array([min(qpoints), max(qpoints)])

        return cls(qpoints01, qpoints12, phis01, phis12, phi0, phi1, phi2)


if __name__ == "__main__":
    steve = DFOKernel.create_from_distribution(distribution = lambda x:(x-0.5)**3, support=(0.2,1.32), n_QP=20)
   
    # qpoints = np.linspace(0.5, 1.5, 20)
    # qvalues = qpoints ** 2
    # carsten = DFOKernel(qpoints=qpoints, qvalues=qvalues)
    
    for francis in [steve]: 
        plt.scatter(francis.alphas01, francis.phis01)
        plt.scatter(francis.alphas12, francis.phis12)     
        plt.scatter(0, francis.phi0)
        plt.scatter(1, francis.phi1)
        plt.scatter(2, francis.phi2)
        plt.show()
    