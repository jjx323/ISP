

#import geneData
#import Ex1

        
expre = 'cos(x[0]) + sin(x[1])'
fun = fe.interpolate(fe.Expression(expre, degree=3), Vreal)
#funL = fe.project(fe.div(fe.grad(fun)), Vreal)
    
expr = 'dd < x[0] && x[0] < xx-dd && dd < x[1] && x[1] < yy-dd ? 1 : 0 '
ce = fe.interpolate(fe.Expression(expr, degree=3, dd=0.05, \
                    xx=2, yy=2), Vreal)
#a_trial, a_test = fe.TrialFunction(Vreal), fe.TestFunction(Vreal)
#funL = fe.Function(Vreal)
#M = fe.assemble(fe.inner(a_trial, a_test)*fe.dx)
#L = fe.assemble(-fe.inner(ce*fe.nabla_grad(q_fun), fe.nabla_grad(a_test))*fe.dx)
#def boundary(x, on_boundary):
#    return on_boundary
#bc = fe.DirichletBC(Vreal, fe.Constant(0.0), boundary)
##bc.apply(M, L)
#fe.solve(M, funL.vector(), L)

funL = fe.project(ce*fe.div(fe.grad(fun)), Vreal)
       
#er = fe.assemble(fe.inner(fun + funL, fun + funL)*fe.dx)
#er1 = fe.assemble(fe.inner(fun, fun)*fe.dx)
#print(er)
#print(er/er1*100)        

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
fe.plot(fun)
plt.subplot(1, 2, 2)
fe.plot(funL)
plt.show()
    
s = fe.Function(Vreal)