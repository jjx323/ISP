import numpy as np 
import matplotlib.pyplot as plt 
import fenics as fe 
import time

from HSolver import *
from InvFun import *

"""
Generate data by Helmholtz solver with true scatterer
"""

# -----------------------------------------------------------------------------
domain_para = {'nx': 200, 'ny': 200, 'dPML': 0.15, 'xx': 2.0, 'yy': 2.0, \
				'sig0': 1.5, 'p': 2.3}

domain = Domain(domain_para)
domain.geneMesh()

equ_para = {'kappa': 5, 'theta': 0.0}

theta_all = np.linspace(0, 2*np.pi, 10)
kappa_all = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
NN = len(theta_all)*len(kappa_all)
NS = 400    # number of measuring points
points = genePoints(NS, 'full', domain_para)
measure = Sample(points)
solR_all, solI_all = np.zeros((NS, NN)), np.zeros((NS, NN))

# specify the forward solver and function space of the coefficients
Fsol = Helmholtz(domain, equ_para)
Vreal = Fsol.getFunctionSpace('real')

# sepcify the true scatterer function
qFunStr = '((0.5 <= x[0] && x[0] <= 1.5 && 0.5 <= x[1] && x[1] <= 1.5) ? 1 : 0)'
#qFunStr = 'exp(-(pow(x[0]-1, 2) + pow(x[1]-1, 2)))'
q_fun = fe.interpolate(trueScatterer(qFunStr, 3), Vreal)
fR = fe.interpolate(fe.Constant(0.0), Vreal)
fI = fe.interpolate(fe.Constant(0.0), Vreal)

iter_num = 0
for kk in range(len(kappa_all)):
    for th in range(len(theta_all)):
        equ_para['theta'], equ_para['kappa'] = theta_all[th], kappa_all[kk]
        exp1 = 'cos(kappa*(x[0]*cos(theta)+x[1]*sin(theta)))'
        exp2 = 'sin(kappa*(x[0]*cos(theta)+x[1]*sin(theta)))'
        uincR = fe.interpolate(fe.Expression(exp1, degree=3, kappa=equ_para['kappa'], \
                                             theta=equ_para['theta']), Vreal)
        uincI = fe.interpolate(fe.Expression(exp2, degree=3, kappa=equ_para['kappa'], \
                                             theta=equ_para['theta']), Vreal)

        fR.vector()[:] = -(equ_para['kappa']**2)*q_fun.vector()[:]*uincR.vector()[:]
        fI.vector()[:] = -(equ_para['kappa']**2)*q_fun.vector()[:]*uincI.vector()[:]
        
        #start = time.time()
        Fsol.geneForwardMatrix('full', q_fun, fR, fI)
        Fsol.solve()
        #end = time.time()
        #print(end-start)
        
        solR_all[:, iter_num] = measure.sampling(Fsol.uReal) 
        solI_all[:, iter_num] = measure.sampling(Fsol.uImag)
        # track the iteration
        iter_num += 1
        print('Iterate ', iter_num, ' steps')

# -----------------------------------------------------------------------------
#from multiprocessing import Pool
#
#domain_para = {'nx': 200, 'ny': 200, 'dPML': 0.15, 'xx': 2.0, 'yy': 2.0, \
#				'sig0': 1.5, 'p': 2.3}
#
#domain = Domain(domain_para)
#domain.geneMesh()
#
#equ_para = {'kappa': 5, 'theta': 0.0}
#
#theta_all = np.linspace(0, 2*np.pi, 10)
#kappa_all = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
#NN = len(theta_all)*len(kappa_all)
#points = []
#NS = 100    # number of measuring points
#for po in np.linspace(0.1, 1.9, NS):
#    points.append((domain_para['yy'], po))
#measure = Sample(points)
#solR_all, solI_all = np.zeros((NS, NN)), np.zeros((NS, NN))
## sepcify the true scatterer function
#qFunStr = '((0.5 <= x[0] && x[0] <= 1.5 && 0.5 <= x[1] && x[1] <= 1.5) ? 1 : 0)'
#q_fun = trueScatterer(qFunStr, 3)
#
#def paral(th):
#    equ_para['theta'], equ_para['kappa'] = theta_all[th], kappa_all[kk]
#    uincR = fe.Expression('cos(kappa*(x[0]*cos(theta)+x[1]*sin(theta)))', degree=3, \
#    						kappa=equ_para['kappa'], theta=equ_para['theta'])
#    uincI = fe.Expression('sin(kappa*(x[0]*cos(theta)+x[1]*sin(theta)))', degree=3, \
#    						kappa=equ_para['kappa'], theta=equ_para['theta'])
#        
#    fR = -equ_para['kappa']*equ_para['kappa']*q_fun*uincR
#    fI = -equ_para['kappa']*equ_para['kappa']*q_fun*uincI
#        
#    Fsol = Helmholtz(domain, equ_para)
#    Fsol.geneForwardMatrix(q_fun, fR, fI)
#    Fsol.solve()
#    solR_all[:, kk+th] = measure.sampling(Fsol.uReal) 
#    solI_all[:, kk+th] = measure.sampling(Fsol.uImag)
#
#for kk in range(len(kappa_all)):
#    with Pool(processes=5) as p:
#        p.map(paral, range(len(theta_all)), chunksize=2)
#    print(kk)
#    

# -----------------------------------------------------------------------------
# solR_all has been saved in the file of dataR, with solutions of order 
# frequency 1, angle 1, angle 2, ... , angle N, 
# frequency 2, angle 1, angle 2, ... , angle N, ......
np.save('dataR', solR_all)
np.save('dataI', solI_all)

# -----------------------------------------------------------------------------

