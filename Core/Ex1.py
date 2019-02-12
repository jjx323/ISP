import numpy as np 
import matplotlib.pyplot as plt 
import fenics as fe 

from HSolver import *

plt.close()

# load the measuring data
uRT = np.load('dataR.')
uIT = np.load('dataI.')

# specify basic parameters
domain_para = {'nx': 150, 'ny': 150, 'dPML': 0.15, 'xx': 2.0, 'yy': 2.0, \
				'sig0': 1.5, 'p': 2.3}

domain = Domain(domain_para)
domain.geneMesh()

equ_para = {'kappa': 5, 'theta': 0.0}
uincR = fe.Expression('cos(kappa*(x[0]*cos(theta)+x[1]*sin(theta)))', degree=3, \
	kappa=equ_para['kappa'], theta=equ_para['theta'])
uincI = fe.Expression('sin(kappa*(x[0]*cos(theta)+x[1]*sin(theta)))', degree=3, \
	kappa=equ_para['kappa'], theta=equ_para['theta'])

q_fun = fe.Expression('0.0', degree=3)

kappa = equ_para['kappa']
fR = -kappa*kappa*q_fun*uincR
fI = -kappa*kappa*q_fun*uincI

Fsol = Helmholtz(domain, equ_para)
Fsol.geneForwardMatrix(q_fun, fR, fI)
Fsol.solve()
Fsol.drawSolution()
