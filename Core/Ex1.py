import numpy as np 
import matplotlib.pyplot as plt 
import fenics as fe 

from HSolver import *
from InvFun import *

plt.close()

# load the measuring data
uRT = np.load('dataR.npy')
uIT = np.load('dataI.npy')

# specify basic parameters
domain_para = {'nx': 150, 'ny': 150, 'dPML': 0.15, 'xx': 2.0, 'yy': 2.0, \
				'sig0': 1.5, 'p': 2.3}

domain = Domain(domain_para)
domain.geneMesh()

theta_all = np.linspace(0, 2*np.pi, 10)
kappa_all = [1.0] #2.0, 3.0, 4.0, 5.0, 6.0]
Ntheta, Nkappa = len(theta_all), len(kappa_all)
NN = Ntheta*Nkappa
NS = 400    # number of measuring points
points = genePoints(NS, 'full', domain_para)
measure = Sample(points)
equ_para = {'kappa': 0.0, 'theta': 0.0}

# init the scatterer
Fsol = Helmholtz(domain, equ_para)
Asol = Helmholtz(domain, equ_para)
Vreal = Fsol.getFunctionSpace('real')
q_fun = initScatterer(Vreal, 'Zero')

# specify the regularization term
reg = Regu('L2_1')

# loop for inversion
iter_num = 0
flag = 'full'   # assemble with all of the coefficients
for freIndx in range(Nkappa):  # loop for frequencies
    for angIndx in range(Ntheta):   # loop for incident angle
        equ_para['kappa'], equ_para['theta'] = kappa_all[freIndx], theta_all[angIndx]
        # ---------------------------------------------------------------------
        # solve forward problem 
        # init the source function
        uincR = fe.Expression('cos(kappa*(x[0]*cos(theta)+x[1]*sin(theta)))', degree=3, \
	                          kappa=equ_para['kappa'], theta=equ_para['theta'])
        uincI = fe.Expression('sin(kappa*(x[0]*cos(theta)+x[1]*sin(theta)))', degree=3, \
	                          kappa=equ_para['kappa'], theta=equ_para['theta'])
        fR = -(equ_para['kappa']**2)*q_fun*uincR
        fI = -(equ_para['kappa']**2)*q_fun*uincI
        # solve equation
        Fsol.geneForwardMatrix(flag, q_fun, fR, fI)
        #start = time.time()
        Fsol.solve()
        uR, uI = Fsol.uReal, Fsol.uImag
        #end = time.time()
        #print('1: ', end-start)
        # ---------------------------------------------------------------------
        # calculate residual
        uRM = measure.sampling(Fsol.uReal)
        uIM = measure.sampling(Fsol.uImag)
        resR = uRT[:, freIndx+angIndx] - uRM
        resI = uIT[:, freIndx+angIndx] - uIM
        # ---------------------------------------------------------------------
        # solve adjoint problem
        # init the source function
        Asol.geneForwardMatrix(flag, q_fun)  # generate matrixes with source equal to zero
        # add point source
        magnitudeR = -(equ_para['kappa']**2)*resR
        magnitudeI = -(equ_para['kappa']**2)*(-resI)
        Asol.addPointSourceR(points, magnitudeR)
        Asol.addPointSourceI(points, magnitudeI)
        # solve equation
        #start = time.time()
        Asol.solve()
        uaR, uaI = Asol.uReal, -Asol.uImag
        #end = time.time()
        #print('2: ', end-start)
        # ---------------------------------------------------------------------
        # calculate the gradient and update the scatterer
        uincR_ = fe.interpolate(uincR, Vreal)
        uincI_ = fe.interpolate(uincI, Vreal)
        cR, cI = Fsol.get_s1s2(Vreal)
        Fdq = (uincR_ + cR*uR - cI*uI)*uaR + (uincI_ + cR*uI + cI*uR)*uaI
        q_fun = q_fun + fe.Constant(0.001)*Fdq
        # only assemble coefficients concerned with q_fun
        flag = 'simple'
        # track the iteration
        iter_num += 1
        print('Iterate ', iter_num, ' steps')

# postpossing      
plt.figure()
fe.plot(q_fun)
plt.show()
        


