import numpy as np 
import matplotlib.pyplot as plt 
import fenics as fe 
import time

from Core.HSolver import *
from Core.InvFun import *
from Core.AddNoise import *

plt.close()
# load the measuring data
# [sol_all, theta_all, kappa_all, qStrT]
dAR = np.load('/home/jjx323/Projects/ISP/Data/dataRc.npy')
dAI = np.load('/home/jjx323/Projects/ISP/Data/dataIc.npy')
uRT, uIT = dAR[0], dAI[0]
theta_all, kappa_all = dAR[1], dAR[2]
qStrT = dAR[3]

# add noise to the data
shapeU = uRT.shape
for i in range(shapeU[1]):
    uRT[:,i], _ = addGaussianNoise(uRT[:,i], {'noise_level': 0.5, 'rate': 1})
    uIT[:,i], _ = addGaussianNoise(uIT[:,i], {'noise_level': 0.5, 'rate': 1})

# -----------------------------------------------------------------------------  
# specify basic parameters
domain_para = {'nx': 120, 'ny': 120, 'dPML': 0.15, 'xx': 2.0, 'yy': 2.0, \
				'sig0': 1.5, 'p': 2.3}

domain = Domain(domain_para)
domain.geneMesh()
#
Ntheta, Nkappa = len(theta_all), len(kappa_all)
NN = Ntheta*Nkappa
NS = 400    # number of measuring points
points = genePoints(NS, 'full', domain_para)
measure = Sample(points)
equ_para = {'kappa': 0.0, 'theta': 0.0}

# init the scatterer
Fsol = Helmholtz(domain, equ_para)
Asol = Helmholtz(domain, equ_para)
Vreal, order = Fsol.getFunctionSpace('real')

# specify the true scatterer for test
q_funT = fe.interpolate(trueScatterer(qStrT, 3), Vreal)
eng2 = fe.assemble(fe.inner(q_funT, q_funT)*fe.dx)
# init the scatterer 
q_fun = initScatterer(Vreal, 'Zero')
# init the force term
fR = fe.interpolate(fe.Constant(0.0), Vreal)
fI = fe.interpolate(fe.Constant(0.0), Vreal)

# specify the regularization term
reg = Regu('L2_1')
gamma = 0.001   # regularization parameter

drawF = 'False'
error_all, q_fun_all = [], []
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
        Fsol.geneForwardMatrix(flag, q_fun, equ_para['kappa'], fR, fI)
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
        Asol.geneForwardMatrix(flag, q_fun, equ_para['kappa'])  # fR and fI is zero
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
        # add the regularization term
        regrid = reg.evaGrad(q_fun, Vreal)
        Fdqv = Fdqv + gamma*regrid.vector()[:]
        # update the scatterer
        q_fun.vector()[:] = q_fun.vector()[:] + 0.01*Fdqv
        #end = time.time()
        #print('3: ', end-start)
        # only assemble coefficients concerned with q_fun
        flag = 'simple'
        # track the iteration
        iter_num += 1
        print('kappa = {:2}; angle = {:3.2f}; iter_num = {:3}'.format(equ_para['kappa'], \
              equ_para['theta'], iter_num))
    # evaluation of the error
    eng1 = fe.assemble(fe.inner(q_funT-q_fun, q_funT-q_fun)*fe.dx)
    error_temp = eng1/eng2
    error_all.append(error_temp)
    print('L2 norm error is {:.2f}%'.format(error_temp*100))  
    # draw and save the intermediate inversion results
    if drawF == 'True':
        plt.figure()
        fig1 = fe.plot(q_fun)
        plt.colorbar(fig1)
        expN = '/home/jjx323/Projects/ISP/ResultsFig/invQ' + str(iter_num) + '.eps'
        plt.savefig(expN, dpi=150)
        plt.close()
    q_fun_all.append(q_fun.vector()[:])

# -----------------------------------------------------------------------------
# postprocessing      
fig2 = my_draw3D(q_fun, [0, 2, 0, 2])
#plt.close()
# save inversion results
vtkfile = fe.File('/home/jjx323/Projects/ISP/ResultsFig/q_fun_c.pvd')
vtkfile << q_fun
# save matrix and reconstruct info
np.save('/home/jjx323/Projects/ISP/Results/q_fun_vector_c', [q_fun.vector()[:], \
        domain_para, ['P', order], q_fun_all])
# error
plt.figure()
plt.plot(error_all)
plt.show()
print('The final L2 norm error is {:.2f}%'.format(error_all[-1]*100))  
np.save('/home/jjx323/Projects/ISP/Results/errorAll_c', error_all)      


