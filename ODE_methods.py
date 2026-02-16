# All possible SDE approximation schemes are collected here. 
# Their input and output has the same general strucute:
# INPUT: - V (batch_size/N x d)-tensor-matrix): set of d-dimensional agents/particles (of the batch)
#        - V_alpha ((1xd)-tensor-matrix): consensus point of all agents 
#        - l (scalar): drift parameter lambda
#        - dt (scalar): step size of one iteration (T/M in the standard setting)
#        - device (device): represents where torch.tensors will be allocated
# OUTPUT: - V_prime ((batch_size/N x d)-tensor-vector): 
# If any scheme relies on more input variables, we will note that at the specific function of the scheme


import torch


# Euler scheme for ODEs
def update_Euler(V, V_alpha, l, dt, device=None):
    device = torch.device('cpu') if device is None else device
    with torch.no_grad():  # disables gradient calculation (reduces memory consumption - cannot use .backward() anymore)
        V -= l * (V - V_alpha) * dt  # compute update
    return V




# Linear Steklov Method
# INPUT: - function (function): optimization function f
#        - alpha (scalar): agent weight for CP
def update_linear_Steklov(V, V_alpha, l, dt, function, alpha , device=None):
    device = torch.device('cpu') if device is None else device
    with torch.no_grad():
        energy_values = function(V).to(device)
        weights = torch.exp(-alpha * (energy_values - energy_values.min())).reshape(-1, 1).to(device)  # compute the weights
        exp_factor = torch.exp(-l * dt * (1 - weights))  # compute exponential drift coefficient

        coeff = torch.zeros_like(weights)  # coefficient in front of (ν_f^α(Z_k) - a_n Z_k)
        mask_not_one = torch.abs(weights - 1.0) > 1e-8  # case: a_n != 1 (up to tolerance)
        mask_is_one = ~mask_not_one  # case: a_n ≈ 1
        coeff[mask_not_one] = (1 - exp_factor[mask_not_one]) / (1 - weights[mask_not_one])  # case a_n != 1
        coeff[mask_is_one] = dt  # case a_n = 1

        V = exp_factor * V + coeff * (V_alpha - weights * V)  # compute update
    return V




# tamed Euler scheme for ODEs
# INPUT: - epoch (integer): iteration step m od discretization
def update_tamed_Euler(V, V_alpha, l, dt, epoch, device=None):
    device = torch.device('cpu') if device is None else device
    with torch.no_grad():  # disables gradient calculation (reduces memory consumption - cannot use .backward() anymore)
        diff = V - V_alpha  # calculate deviation of agents to CP (Nxd)-tensor-matrix
        V -= l * diff / (1.0 + (epoch+1)**(-1/2) * l * torch.abs(diff)) * dt  # compute update
    return V




# Simplified Theta Method (linearized implicit theta)
def update_Theta(V, V_alpha, l, dt, theta, device=None):
    device = torch.device('cpu') if device is None else device
    with torch.no_grad():
        V = (V - (1 - theta) * l * V * dt  + l * dt * V_alpha) / (1 + theta * l * dt) # compute update
    return V




# Semi Heun Method
# INPUT: - function (function): optimization function f
#        - alpha (scalar): agent weight for CP
def update_semi_Heun(V, V_alpha, l, dt, function, alpha, device=None):
    device = torch.device('cpu') if device is None else device
    with torch.no_grad():
        # Predictor (explicit Euler step)
        diff_Euler = V - V_alpha
        V_tilde = V - l * diff_Euler * dt

        # Compute new consensus point
        energy_values = function(V_tilde).to(device)
        V_alpha_tilde = compute_v_alpha(energy_values, V_tilde, alpha, device=device)

        # Corrector: average of old and new drift
        diff_tilde = V_tilde - V_alpha_tilde
        V -= 0.5 * l * (diff_Euler + diff_tilde) * dt
    return V












#####################################################################


# functions used by Heun
def compute_v_alpha(energy_values, agents, alpha, device=None):
    device = torch.device('cpu') if device is None else device
    weights = torch.exp(-alpha * (energy_values - energy_values.min())).reshape(-1, 1).to(device)  # turn vector in (batch_size x 1)-tensor-matrix
    consensus = (weights * agents) / weights.sum()  # calculate the weighted agents (differently weighted in each dimension)
    return consensus.sum(dim=0)  # sum all weighted agents together (in each dimension)

def compute_energy_values(function, agents, device=None):
    device = torch.device('cpu') if device is None else device
    return function(agents).to(device)  # function evaluation of each agent in V recorded in device