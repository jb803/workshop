"""Demo for finding low Reynolds number laminar flows with FE"""
import math

#1) Import FEniCS (dolfin)
from dolfin import *

#2) Load the mesh
mesh = Mesh('mesh.xml')

#3) Define the function elements
P2      = FiniteElement('CG',mesh.ufl_cell(),2) #Quadratic for the velocity components
P1      = FiniteElement('CG',mesh.ufl_cell(),1) #Linear for the pressure components

Fs          = FunctionSpace(mesh,MixedElement([P2,P2,P1])) #u,v,p
visSpace    = FunctionSpace(mesh,MixedElement([P2,P2]))    #u,v   #For visualising

visFunc     = Function(visSpace)    #For visualising
visFunc.rename('u','u')             #For visualising
#4) Define the state vector, the test and trial functions and the resiudal

state   = Function(Fs)
test    = TestFunction(Fs)
trial   = TrialFunction(Fs)

#5) Split the test and state into their components
(u,v,p)     = split(state)
(a,b,q)     = split(test)
(du,dv,dp)  = split(trial)

#6) Define the redisual and Jacobian
Re  = Constant(50.)


R   = ( a*( u*u.dx(0) + v*u.dx(1))
       +b*( u*v.dx(0) + v*v.dx(1))
       -p*( a.dx(0)  + b.dx(1))     #Integrating the pressure term by parts
       +(1./Re)*( inner(grad(a),grad(u)) + inner(grad(b),grad(v))) 
       +q*(u.dx(0) + v.dx(1))       #Continuity
      )*dx


#7) Define the domain boundaries
def inlet(x,on_boundary):           #inlet  is x=-8
    return near(x[0],-8) and on_boundary
def slip(x,on_boundary):            #The slip walls are at y = +-4
    return near(abs(x[1]),8) and on_boundary
def noslip(x,on_boundary):
    return x[0]**2. + x[1]**2. < 1.1 and on_boundary

#8) Define the boundary conditions
bcIter1     = [DirichletBC(Fs.sub(0), 1.0, inlet),
               DirichletBC(Fs.sub(1),   0, inlet),
               DirichletBC(Fs.sub(0),   0, noslip),
               DirichletBC(Fs.sub(1),   0, noslip),
               DirichletBC(Fs.sub(1),   0, slip)]
bcIter      = [DirichletBC(Fs.sub(0),   0, inlet),
               DirichletBC(Fs.sub(1),   0, inlet),
               DirichletBC(Fs.sub(0),   0, noslip),
               DirichletBC(Fs.sub(1),   0, noslip),
               DirichletBC(Fs.sub(1),   0, slip)]

#9) We get the jacobian matrix
Jac     = derivative(R,state,du=trial)

#10) We solve by a Newton method
maxIter = 10
tol     = 1e-12
i       = 0
delta   = Function(Fs)

visFile = File('baseflow.pvd')

while i < maxIter:
    #We select the appropriate boundary conditions
    if i == 0:
        bc = bcIter1
    else:
        bc = bcIter
        
    solve(Jac == -R, delta, bc)
    
    #We update the degrees of freedom
    state.vector()[:] += delta.vector()
    
    (u,v,p) = state.split(True)     #For visualising
    assign(visFunc,[u,v])           #For visualising
    
    visFile<<visFunc
    
    deltaMag    = delta.vector().norm('l2')
    
    print('At iteration %d the error was: %g'%(i,deltaMag))
    #We check how much the state vector changed by (NOTE - this is a bad way of checking convergence but avoids having to explicitly create PETSc matrices)
    if delta.vector().norm('l2')< tol:
        break
        
    i += 1
    

"""
#Adjoint demo

#We want to find the sensitivity of drag ( via energy dissipation proxy) to momentum injection
#We get the matrix form of the Jacobian and transpose it
AA = PETScMatrix()
assemble(adjoint(Jac),tensor=AA)



#We assemble the source term
bb = PETScVector()
assemble( -(1./Re)*( inner(grad(u),grad(a)) + inner(grad(v),grad(b)))*dx, tensor=bb) #The -ve sign is here because we want to DECREASE the drag

adjoint = Function(Fs)

for bc in bcIter:
    bc.apply(AA)
    bc.apply(bb)

solve(AA,adjoint.vector(),bb,'mumps')

(u,v,p) = adjoint.split(True)
assign(visFunc,[u,v])

File('adjoint.pvd')<<visFunc
"""



