from scipy.optimize import fsolve 

def computes_alpha_beta_residuals(variables, *args):
    (alpha,beta)= variables
    (E, nu, p) = args
    
    bulk = E / (3*(1 - 2 * nu)) 
    mu = E / (2 * (1 + nu))
    eqn_1 = mu * alpha - mu * 1 / alpha + bulk * beta**2 * (alpha* beta**2 -1) - (- p) # compression is positive
    eqn_2 = beta * mu  - mu * 1 / beta + bulk * beta * (alpha**2 * beta**2 - alpha)  
    return [eqn_1,eqn_2]

def computes_displacements(x,y,z, alpha, beta):
    U = ((alpha-1)*x, (beta-1)*y, (beta-1)*z)
    return U 

def compute_analytic_solution(x, y, z, *data):
    (E, nu, pressure) = data
    alpha_0 = 0.05
    beta_0 = 0.05 

    alpha, beta = fsolve(computes_alpha_beta_residuals, [alpha_0, beta_0], args = data)
    #print("alpha is: %.2f, beta is: %.2f" % (alpha, beta))
    
    ux, uy, uz = computes_displacements(x, y, z, alpha, beta)
    return ux, uy, uz

# From main
if __name__ == '__main__':

    sample_8220 = (1.9,1,1,1.3,.3,3.0,-1.443487,0.224480,0.224480)
    Lx, Ly, Lz, E, nu, p, Ux_train, Uy_train, Uz_train = sample_8220

    ux, uy, uz = compute_analytic_solution(Lx,Ly,Lz, E, nu, p)
    
    # Print ux, uy and uz with text
    print("ux_train is: %.5f m, uy_train is: %.5f m, uz_train is: %.5f m" % (Ux_train, Uy_train, Uz_train))
    print("ux_test is: %.5f m, uy_test is: %.5f m, uz_test is: %.5f m" % (ux, uy, uz))