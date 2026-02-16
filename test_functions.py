# This file contains the optimization functions f : R^d --> R, which are used to test the different SDE methods.
# Each function has the same structure:
# INPUT : - V ((N x D)-torch-matrix): describing the data, which will be evaluated
# OUTPUT : - V_prime ((1xN)-torch-matrix): describes the function evaluation of each data point
# If any function relies on more input variables, we will note that at the specific function

import torch



# Rastrigin function
def rastrigin(V):
    return rastrigin_c()(V)

def rastrigin_c(c=10):
    # INPUT: - c (scalar): sclaes the problem (mostly it is set to 10)
    return lambda V: (V ** 2 - c * torch.cos(2 * torch.pi * V)).sum(dim=1) + V.size()[1] * c


# Square function
def square(V):
    return V ** 2


# Mishra's Bird function
def mishras_bird(V):
    if V.size()[1] == 2:
        return torch.sin(V[:,1]) * torch.exp((1-torch.cos(V[:,0]))**2) + torch.cos(V[:,0]) * torch.exp((1-torch.sin(V[:,1]))**2) + (V[:,0]-V[:,1])**2
    else:
        return "ERROR: The input vectors are not 2-dimensional."
    

# McCormick function
def mc_cormick(V):
    if V.size()[1] == 2:
        return torch.sin(V[:,0]+V[:,1]) + (V[:,0]-V[:,1])**2 * - 1.5*V[:,0] + 2.5*V[:,1] + 1
    else:
        return "ERROR: The input vectors are not 2-dimensional."
    

# Hölder table function
def hoelder(V):
    if V.size()[1] == 2:
        return - torch.abs(torch.sin(V[:,0]) * torch.cos(V[:,1]) * torch.exp(torch.abs(1-torch.sqrt(V[:,0]**2+V[:,1]**2)/torch.pi)))
    else:
        return "ERROR: The input vectors are not 2-dimensional."
    

# Rosenbrock function
def rosenbrock(V, a=1, b=100):
    return ((a - V[:, :-1]) ** 2 + b * (V[:, 1:] - V[:, :-1] ** 2) ** 2).sum(axis=1)

# Himmelblau function
def himmelblau(V):
    if V.size()[1] == 2:
        return (V[:, 0]**2 + V[:, 1] - 11)**2 + (V[:, 0] + V[:, 1]**2 - 7)**2
    else:
        return "ERROR: The input vectors are not 2-dimensional."
    

# Bazaraa-Shetty function
def bazaraa_shetty(V):
    if V.size()[1] == 2:
        return (V[:, 0] - 2)**4 + (V[:, 0] - 2*V[:, 1])**2
    else:
        return "ERROR: The input vectors are not 2-dimensional."



# Discontinuous Exponential Function
def discontinuous_exp(V, a=None, u=None):
    N, d = V.size()

    # Set default values if not provided
    if a is None:
        a = 5 * torch.ones(d)
    if u is None:
        u = 0.5 * torch.ones(d)

    if V.size()[1] != d:
        return "ERROR: Input must be d-dimensional."

    # Mask: True if the point lies within the bounds (all x_i <= u_i)
    mask = (V <= u).all(dim=1)

    # Apply summation formula
    linear_comb = - torch.exp((V @ a))

    # Initialize result
    result = torch.zeros(N)
    result[mask] = linear_comb[mask]

    return result
