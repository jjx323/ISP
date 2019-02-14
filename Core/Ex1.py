import numpy as np 
import matplotlib.pyplot as plt 
import fenics as fe 
import time

from HSolver import *
from InvFun import *

plt.close()
# load the measuring data
uRT = np.load('dataR.npy')
uIT = np.load('dataI.npy')

# specify basic parameters
domain_para = {'nx': 120, 'ny': 120, 'dPML': 0.15, 'xx': 2.0, 'yy': 2.0, \
				'sig0': 1.5, 'p': 2.3}

domain = Domain(domain_para)
domain.geneMesh()

theta_all = np.linspace(0, 2*np.pi, 10)
kappa_all = [1.0, 2.0, 3.0]#, 4.0, 5.0, 6.0]
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
#qFunStr = '((0.5 <= x[0] && x[0] <= 1.5 && 0.5 <= x[1] && x[1] <= 1.5) ? 1 : 0)'
#q_fun = fe.interpolate(trueScatterer(qFunStr, 3), Vreal)
q_fun = initScatterer(Vreal, 'Zero')
fR = fe.interpolate(fe.Constant(0.0), Vreal)
fI = fe.interpolate(fe.Constant(0.0), Vreal)

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
        exp1 = 'cos(kappa*(x[0]*cos(theta)+x[1]*sin(theta)))'
        uincR = fe.interpolate(fe.Expression(exp1, degree=3, kappa=equ_para['kappa'], \
                                             theta=equ_para['theta']), Vreal)
        exp2 = 'sin(kappa*(x[0]*cos(theta)+x[1]*sin(theta)))'
        uincI = fe.interpolate(fe.Expression(exp2, degree=3, kappa=equ_para['kappa'], \
                                             theta=equ_para['theta']), Vreal)
        fR.vector()[:] = -(equ_para['kappa']**2)*q_fun.vector()[:]*uincR.vector()[:]
        fI.vector()[:] = -(equ_para['kappa']**2)*q_fun.vector()[:]*uincI.vector()[:]
        # solve equation
        Fsol.geneForwardMatrix(flag, q_fun, fR, fI)
        #start = time.time()
        Fsol.solve()
        uR, uI = fe.interpolate(Fsol.uReal, Vreal), fe.interpolate(Fsol.uImag, Vreal)
        #end = time.time()
        #print('1: ', end-start)
        # ---------------------------------------------------------------------
        # calculate residual
        uRM = measure.sampling(Fsol.uReal)
        uIM = measure.sampling(Fsol.uImag)
        resR = uRT[:, iter_num] - uRM
        resI = uIT[:, iter_num] - uIM
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
        uaR, uaI = fe.interpolate(Asol.uReal, Vreal), fe.interpolate(Asol.uImag, Vreal)
        uaI.vector()[:] = -uaI.vector()[:]
        #end = time.time()
        #print('2: ', end-start)
        # ---------------------------------------------------------------------
        # calculate the gradient and update the scatterer
        #start = time.time()
        cR, cI = Fsol.get_s1s2(Vreal, 'vector')
        Fdqv = (uincR.vector()[:] + cR*uR.vector()[:] - \
                cI*uI.vector()[:])*uaR.vector()[:] + \
                (uincI.vector()[:] + cR*uI.vector()[:] + \
                 cI*uR.vector()[:])*uaI.vector()[:]    
        q_fun.vector()[:] = q_fun.vector()[:] + 0.01*Fdqv
        #end = time.time()
        #print('3: ', end-start)
        # only assemble coefficients concerned with q_fun
        flag = 'simple'
        # track the iteration
        iter_num += 1
        print('Iterate ', iter_num, ' steps')

# postprocessing      
# draw the 
plt.figure()
fig1 = fe.plot(q_fun)
plt.colorbar(fig1)
#plt.savefig('invQ1.eps', dpi=150)
#fig2 = my_draw3D(q_fun, [2, 2])
#plt.close()
# save inversion results
vtkfile = fe.File('q_fun.pvd')
vtkfile << q_fun
# evaluation of the error
qFunStr = '((0.5 <= x[0] && x[0] <= 1.5 && 0.5 <= x[1] && x[1] <= 1.5) ? 1 : 0)'
q_funT = fe.interpolate(trueScatterer(qFunStr, 3), Vreal)
eng1 = fe.assemble(fe.inner(q_funT-q_fun, q_funT-q_fun)*fe.dx)
eng2 = fe.assemble(fe.inner(q_funT, q_funT)*fe.dx)
error = eng1/eng2
print('L2 norm error is {:.2f}%'.format(error*100))        


