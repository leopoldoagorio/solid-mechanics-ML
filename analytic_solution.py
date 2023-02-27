from scipy.optimize import fsolve 

def analytic_solver(variables):
    (alpha,beta)= variables
    bulk = E / (3*(1-2*nu)) 
    mu = E / (2*(1+nu))
    eqn_1 = mu * alpha - mu * 1 / alpha + bulk * beta**2 * (alpha* beta**2 -1) - pressure
    eqn_2 = beta * mu  - mu * 1 / beta + bulk * beta * (alpha**2 * beta**2 - alpha)  
    return [eqn_1,eqn_2]

def displacements(x,y,z, alpha, beta):
    U = ((alpha-1)*x, (beta-1)*y, (beta-1)*z)
    return U 

# From main
if __name__ == '__main__':
    E = 2.
    pressure = 1.
    nu = .3
    Lx = 3 
    Ly = 1
    Lz = 1

    alpha, beta = fsolve(analytic_solver, [0.1, 0.1]) 

    ux, uy, uz = displacements(Lx,Ly,Lz, alpha, beta)

    print(ux, uy, uz)