import os
import sys
import importlib
import glob
import time
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.RationalApproximation import RationalApproximation_AAA
from scipy.special import gamma 
import pandas as pd 

class ModesImplicitEuler: 
    def __init__(self, poles, weights, **kwargs):
        self.poles = poles
        self.weights = weights

    # u_k^n+1 = alpha_k u_k^n + beta_k u^n+1 + gamma_k u^n     
    def get_coefficients(self, dt): 
        alphas = 1 / (1 + dt * self.poles)
        betas = (dt * self.weights) / (1 + dt * self.poles)
        gammas = 0 * self.poles

        # print(alphas.shape, betas.shape, gammas.shape)
        return alphas, betas, gammas 

# class ModesCrankNicolson: 

#     def __init__(self, poles, weights, w_inf):
#         self.poles = poles
#         self.weights = weights
#         self.w_inf = w_inf

#     # u_k^n+1 = alpha_k u_k^n + beta_k u^n+1 + gamma_k u^n     
#     def get_coefficients(self,dt): 
#         denom = (1 + 0.5 * dt * self.poles)

#         alphas = (1 - 0.5 * dt * self.poles)   / denom
#         betas  = 0.5 * dt * self.weights / denom 
#         gammas = 0.5 * dt * self.weights / denom 

#         return alphas, betas, gammas  

class ModesExponentialIntegrator: 

    def __init__(self, poles, weights):
        self.poles = poles
        self.weights = weights

    # u_k^n+1 = alpha_k u_k^n + beta_k u^n+1 + gamma_k u^n     
    def get_coefficients(self,dt): 
        alphas = np.exp(- self.poles * dt)
        betas  = self.weights * (alphas - (1 - self.poles * dt)) / (dt * self.poles ** 2)
        gammas = self.weights * (1 - (1 + self.poles * dt) * alphas) / (dt * self.poles ** 2)
        return alphas, betas, gammas  

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
    
# Distribution phi 
@np.vectorize
def Distribution(alpha): 
    return gamma(6 - alpha) / 120

# Right side of the equation 
@np.vectorize
def f(t): 
    if t == 0: 
        return 0
    if t == 1: 
        return 2
    return (t ** 5 - t**3) / np.log(t)

# Analytical solution 
@np.vectorize
def sol(t): 
    return t ** 5

#########################################
### Main  
#########################################

if __name__ == "__main__":

    T  = 1

    flag_convergence_dt = True 
    error_dt = [] 
    dts = [1e-6 * 2 ** (10-i) for i in range(11)]
    print("Minimal time step:", dts[-1])    

    flag_convergence_QF = True 
    error_QF = [] 
    QFs = [2 ** i  for i in range(20)]
    print("Maximal number of quadrature points:", 2*QFs[-1]+3)
    
    #########################################
    ### Convergence Analysis with respect to the QF  
    #########################################

    if flag_convergence_QF: 

        dt = np.min(dts) 
        n_timesteps = int(T/dt) + 1
        t = dt * np.asarray(range(n_timesteps))

        for n_QF in QFs: 

            #########################################
            ### Quadrature of the integral 
            #########################################
            
            # interval length / (n for 0 to 1 + n for 1 to 2 + 3 for 0,1,2)
            QFweight = 2 / (2*n_QF+3)
            quadraturePoints01 = np.arange(0 + 0.5/n_QF,1 + 0.5/n_QF, 1/n_QF)
            quadraturePoints12 = np.arange(1 + 0.5/n_QF,2 + 0.5/n_QF, 1/n_QF)
        
            phis01 = Distribution(quadraturePoints01) * QFweight
            phis12 = Distribution(quadraturePoints12) * QFweight
            phi0 = Distribution(0) * QFweight
            phi1 = Distribution(1) * QFweight
            phi2 = Distribution(2) * QFweight

            MTK01 = MultiTermKernel(phis01,quadraturePoints01)
            MTK12 = MultiTermKernel(phis12,quadraturePoints12-1)

            #########################################
            ### Rational Approximation of the Integral Kernel
            #########################################

            Zmin, Zmax = 1/T,1/dt
            tol = 1.e-11
            MaxDegree = 30
            nSupportPoints = 100

            RA01 = RationalApproximation_AAA(alpha=0.5, tol=tol, MaxDegree=MaxDegree, nSupportPoints=nSupportPoints, Zmin= Zmin, Zmax= Zmax, verbose=False, TargetFunction=MTK01.RATarget)
            RA12 = RationalApproximation_AAA(alpha=0.5, tol=tol, MaxDegree=MaxDegree, nSupportPoints=nSupportPoints, Zmin= Zmin, Zmax= Zmax, verbose=False, TargetFunction=MTK12.RATarget)

            weights01 = RA01.c
            poles01 = RA01.d
            winf01 = RA01.c_inf
            m01 = weights01.shape

            weights12 = RA12.c
            poles12 = RA12.d
            winf12 = RA12.c_inf
            m12 = weights12.shape

            #########################################
            ### Numerical Scheme 
            #########################################

            beta0 = phi0 + np.sum(weights01) - np.sum(weights12 * poles12)
            beta1 = phi1 + np.sum(weights12) + winf01
            beta2 = phi2 + winf12

            modes01 = np.zeros((n_timesteps, m01[0])) 
            modes12 = np.zeros((n_timesteps, m12[0])) 

            u = np.zeros(n_timesteps) 
            v = np.zeros(n_timesteps) 

            # Initial conditions if necessary 
            u[0] = 0
            v[0] = 0 

            TIModes01 = ModesImplicitEuler(poles01, weights01) 
            TIModes12 = ModesImplicitEuler(poles12, weights12) 

            alphas01, betas01, gammas01 = TIModes01.get_coefficients(dt)
            alphas12, betas12, gammas12 = TIModes12.get_coefficients(dt)

            a0 = 0
            a1 = dt 

            print("n_QF:", n_QF)

            # Time loop 
            for n in tqdm(range(n_timesteps-1), ncols=100): 

                # correct 
                A = (- a0 * beta0 - a1 * beta0 
                     + np.sum(a1 * poles01 * (betas01 + gammas01)) 
                     - np.sum(a1 * poles12 ** 2 * (betas12 + gammas12))) / beta2

                # correct 
                B = 1 + (- a0 * beta1 - a0 * a1 * beta0 
                         + np.sum(a0 * a1 * poles01 * betas01) 
                         - np.sum(a0 * a1 * poles12 ** 2 * betas12)) / beta2
                
                # correct  
                C = a1 / beta2 * (- a1 * beta0 - beta1 
                                  + np.sum(a1 * poles01 * betas01) 
                                  - np.sum(a1 * poles12 ** 2 * betas12))

                tmp = (a0 / beta2 * f(n * dt) + a1 / beta2 * f((n+1)*dt) 
                       + A*u[n] + B*v[n] 
                       + np.sum((a0 + a1 * alphas01) * poles01 * modes01[n]) / beta2  
                       - np.sum((a0 + a1 * alphas12) * poles12 **2 * modes12[n]) / beta2)
                
                v[n+1] = tmp / (1-C)
                u[n+1] = u[n] + dt * v[n+1]

                modes01[n+1] = 1 / (1 + dt*poles01) * (modes01[n] + dt * weights01 * u[n+1])
                modes12[n+1] = 1 / (1 + dt*poles12) * (modes12[n] + dt * weights12 * u[n+1])

            #########################################
            ### Compute error 
            #########################################

            t = dt * np.asarray(range(n_timesteps))
            error_QF.append(dt * (np.linalg.norm(sol(t) - u)))

    #########################################
    ### Convergence Analysis with respect to dt
    #########################################
    
    if flag_convergence_dt: 

        n_QF = np.max(QFs)

        #########################################
        ### Quadrature of the integral 
        #########################################
        
        # interval length / (n for 0 to 1 + n for 1 to 2 + 3 for 0,1,2)
        QFweight = 2 / (2*n_QF+3)
        quadraturePoints01 = np.arange(0 + 0.5/n_QF,1 + 0.5/n_QF, 1/n_QF)
        quadraturePoints12 = np.arange(1 + 0.5/n_QF,2 + 0.5/n_QF, 1/n_QF)
    
        phis01 = Distribution(quadraturePoints01) * QFweight
        phis12 = Distribution(quadraturePoints12) * QFweight
        phi0 = Distribution(0) * QFweight
        phi1 = Distribution(1) * QFweight
        phi2 = Distribution(2) * QFweight

        MTK01 = MultiTermKernel(phis01,quadraturePoints01)
        MTK12 = MultiTermKernel(phis12,quadraturePoints12-1)

        for dt in dts: 

            n_timesteps = int(T/dt) + 1
            t = dt * np.asarray(range(n_timesteps))

            #########################################
            ### Rational Approximation of the Integral Kernel
            #########################################

            Zmin, Zmax = 1/T,1/dt
            tol = 1.e-11
            MaxDegree = 30
            nSupportPoints = 100

            RA01 = RationalApproximation_AAA(alpha=0.5, tol=tol, MaxDegree=MaxDegree, nSupportPoints=nSupportPoints, Zmin= Zmin, Zmax= Zmax, verbose=False, TargetFunction=MTK01.RATarget)
            RA12 = RationalApproximation_AAA(alpha=0.5, tol=tol, MaxDegree=MaxDegree, nSupportPoints=nSupportPoints, Zmin= Zmin, Zmax= Zmax, verbose=False, TargetFunction=MTK12.RATarget)

            weights01 = RA01.c
            poles01 = RA01.d
            winf01 = RA01.c_inf
            m01 = weights01.shape

            weights12 = RA12.c
            poles12 = RA12.d
            winf12 = RA12.c_inf
            m12 = weights12.shape

            #########################################
            ### Numerical Scheme 
            #########################################

            beta0 = phi0 + np.sum(weights01) - np.sum(weights12 * poles12)
            beta1 = phi1 + np.sum(weights12) + winf01
            beta2 = phi2 + winf12

            modes01 = np.zeros((n_timesteps, m01[0])) 
            modes12 = np.zeros((n_timesteps, m12[0])) 

            u = np.zeros(n_timesteps) 
            v = np.zeros(n_timesteps) 

            # Initial conditions if necessary 
            u[0] = 0
            v[0] = 0 

            TIModes01 = ModesImplicitEuler(poles01, weights01) 
            TIModes12 = ModesImplicitEuler(poles12, weights12) 

            alphas01, betas01, gammas01 = TIModes01.get_coefficients(dt)
            alphas12, betas12, gammas12 = TIModes12.get_coefficients(dt)

            # Implicit first order time integration 
            a0 = 0
            a1 = dt 

            print("dt:", dt)
            
            # Time loop 
            for n in tqdm(range(n_timesteps-1), ncols=100): 

                # correct 
                A = (- a0 * beta0 
                     - a1 * beta0 
                     + np.sum(a1 * poles01 * (betas01 + gammas01)) 
                     - np.sum(a1 * poles12 ** 2 * (betas12 + gammas12))) / beta2

                # correct 
                B = 1 + (- a0 * beta1 
                         - a0 * a1 * beta0 
                         + np.sum(a0 * a1 * poles01 * betas01) 
                         - np.sum(a0 * a1 * poles12 ** 2 * betas12)) / beta2
                
                # correct  
                C = a1 / beta2 * (- a1 * beta0 
                                  - beta1 
                                  + np.sum(a1 * poles01 * betas01) 
                                  - np.sum(a1 * poles12 ** 2 * betas12))

                tmp = (a0 / beta2 * f(n * dt) 
                    + a1 / beta2 * f((n+1)*dt) 
                    + A * u[n] + B * v[n] 
                    + np.sum((a0 + a1 * alphas01) * poles01 * modes01[n]) / beta2  
                    - np.sum((a0 + a1 * alphas12) * poles12 **2 * modes12[n]) / beta2)
            
                v[n+1] = tmp / (1-C)
                u[n+1] = u[n] + dt * v[n+1]

                modes01[n+1] = alphas01 * modes01[n] + betas01 * u[n+1] + gammas01 * u[n] 
                modes12[n+1] = alphas12 * modes12[n] + betas12 * u[n+1] + gammas12 * u[n] 

            #########################################
            ### Compute error 
            #########################################
            # print(u)
            # plt.loglog(u)
            # plt.show()
            error_dt.append(dt * (np.linalg.norm(sol(t) - u)))

    #########################################
    ### Visualization  
    ##########################################

    if flag_convergence_dt: 
        fig,ax = plt.subplots(figsize = (5,5))
        ax.set_title("Error decay with respect to the time step")
        ax.loglog(dts, error_dt, color = "red", linewidth = 2, marker="o")
        ax.loglog(dts, np.asarray(dts) * np.max(error_dt) / np.max(dts), color = "blue", linewidth = 2, linestyle="--")
        ax.set_xlabel(r'time step size $\Delta t$') 
        ax.set_ylabel(r"Error $\|u_{sol} - u_h\|_2$", fontweight ='bold') 
        plt.tight_layout()
        # fig.savefig("ErrorDecayTimestep.pdf")

    if flag_convergence_QF: 
        fig,ax = plt.subplots(figsize = (5,5))
        ax.set_title("Error decay with respect to the number of quadrature points")
        ax.loglog(QFs,error_QF, color = "blue", linewidth = 2, marker="o")
        ax.set_xlabel(r'number of quadrature points per $[0,1]$') 
        ax.set_ylabel(r"Error $\|u_{sol} - u_h\|_2$", fontweight ='bold') 
        plt.tight_layout()
        # fig.savefig("ErrorDecayQuadraturePoints.pdf")

    plt.show()
