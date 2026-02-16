# All possible SDE approximation schemes are collected here. 
# Their input and output has the same general strucute:
# INPUT: - V (batch_size/N x d)-tensor-matrix): set of d-dimensional agents/particles (of the batch)
#        - V_alpha ((1xd)-tensor-matrix): consensus point of all agents 
#        - anisotropic (): 
#        - l (scalar): drift parameter lambda
#        - sigma (scalar): diffusion parameter
#        - dt (scalar): step size of one iteration (T/M in the standard setting)
#        - device (device): represents where torch.tensors will be allocated
# OUTPUT: - V_prime ((batch_size/N x d)-tensor-vector): 
# If any scheme relies on more input variables, we will note that at the specific function of the scheme


import torch
from utils import inplace_randn


# Euler-Maruyama scheme for SDEs
def cbo_update_Euler_Maruyama(V, V_alpha, anisotropic, l, sigma, dt, device=None):
    device = torch.device('cpu') if device is None else device
    noise = inplace_randn(V.shape, device)
    with torch.no_grad():  # disables gradient calculation (reduces memory consumption - cannot use .backward() anymore)
        diff = V - V_alpha  # calculate deviation for agents to CP to (Nxd)-tensor-matrix
        noise_weight = torch.abs(diff) if anisotropic else torch.norm(diff, p=2, dim=1).reshape(-1,1)  # calculate norm (1xN - since norm on each row) or absolute value (Nxd - bc componentwise multiplication of (Nxd) and (Nxd))
        V -= l * diff * dt  # do deterministic ODE step
        # print(V)
        V += sigma * noise_weight * noise * (dt ** 0.5)  # do stochastic SDE step and weight the noise correctly
        # print(V)
    return V


# tamed Euler scheme for SDEs
# INPUT: - epoch (integer): iteration step m od discretization
def cbo_update_tamed_Euler(V, V_alpha, anisotropic, l, sigma, dt, epoch, device=None):
    device = torch.device('cpu') if device is None else device
    noise = inplace_randn(V.shape, device)
    with torch.no_grad():  # disables gradient calculation (reduces memory consumption - cannot use .backward() anymore)
        diff = V - V_alpha  # calculate deviation fo agents to CP to (Nxd)-tensor-matrix
        noise_weight = torch.abs(diff) if anisotropic else torch.norm(diff, p=2, dim=1).reshape(-1,1)  # calculate norm (1xN - since norm on each row) or absolute value (Nxd - bc componentwise multiplication of (Nxd) and (Nxd))
        V -= l * diff / (1 + (epoch+1)**(-1/2) * torch.abs(l*diff)) * dt  # do deterministic ODE step
        V += sigma * noise_weight * noise * (dt ** 0.5)  # doe stochastic SDE step and weight the noise correctly
    return V


# advanced Heun scheme for SDEs
# INPUT: - function (function): optimization function f
#        - alpha (scalar): agent weight for CP
def cbo_update_advanced_Heun(V, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=None):
    device = torch.device('cpu') if device is None else device
    noise = inplace_randn(V.shape, device)
    with torch.no_grad():  # disables gradient calculation (reduces memory consumption - cannot use .backward() anymore)
        diff_Euler = V - V_alpha  # calculate deviation for agents to CP to (Nxd)-tensor-matrix for Euler step
        noise_weight_Euler = torch.abs(diff_Euler) if anisotropic else torch.norm(diff_Euler, p=2, dim=1).reshape(-1,1)  # calculate norm (1xN - since norm on each row) or absolute value (Nxd - bc componentwise multiplication of (Nxd) and (Nxd)) for Euler step
        V_Euler = - l * diff_Euler * dt + sigma * noise_weight_Euler * noise * (dt ** 0.5)  # calculate X^(star) - V
        batch_energy_values = compute_energy_values(function, V + V_Euler, device=device)  # function evaluation of all agents in the batch
        V_alpha_Heun = compute_v_alpha(batch_energy_values, V + V_Euler, alpha, device=device)  # compute CP for Heun step
        diff_Heun = (V + V_Euler) - V_alpha_Heun  # calculate deviation for agents to CP to (Nxd)-tensor-matrix for Heun step
        noise_weight_Heun = torch.abs(diff_Heun) if anisotropic else torch.norm(diff_Heun, p=2, dim=1)  # calculate norm (1xN - since norm on each row) or absolute value (Nxd - bc componentwise multiplication of (Nxd) and (Nxd))
        V += 1/2 * V_Euler  # do Euler part
        V += 1/2 * (- l * diff_Heun * dt + sigma * noise_weight_Heun * noise * (dt ** 0.5))  # do Heun  part
    return V


# simple Heun scheme for SDEs
# INPUT: - function (function): optimization function f
#        - alpha (scalar): agent weight for CP
def cbo_update_simple_Heun(V, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=None):
    device = torch.device('cpu') if device is None else device
    noise = inplace_randn(V.shape, device)
    with torch.no_grad():  # disables gradient calculation (reduces memory consumption - cannot use .backward() anymore)
        diff_Euler = V - V_alpha  # calculate deviation for agents to CP to (Nxd)-tensor-matrix for Euler step
        noise_weight_Euler = torch.abs(diff_Euler) if anisotropic else torch.norm(diff_Euler, p=2, dim=1).reshape(-1,1)  # calculate norm (1xN - since norm on each row) or absolute value (Nxd - bc componentwise multiplication of (Nxd) and (Nxd)) for Euler step
        noise_Euler = sigma * noise_weight_Euler * noise * (dt ** 0.5) 
        V_Euler = - l * diff_Euler * dt + noise_Euler  # calculate X^(star) - V 
        batch_energy_values = compute_energy_values(function, V + V_Euler, device=device)  # function evaluation of all agents in the batch
        V_alpha_Heun = compute_v_alpha(batch_energy_values, V + V_Euler, alpha, device=device)  # compute CP for Heun step
        diff_Heun = (V + V_Euler) - V_alpha_Heun  # calculate deviation for agents to CP to (Nxd)-tensor-matrix for Heun step
        V += 1/2 * V_Euler + 1/2 * noise_Euler # do Euler part
        V -= 1/2 * l * diff_Heun * dt  # do Heun  part
    return V


# explicit order 1.0 scheme for SDEs
# INPUT: - function (function): optimization function f
#        - alpha (scalar): agent weight for CP
def cbo_update_order_1(V, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=None):
    device = torch.device('cpu') if device is None else device
    noise = inplace_randn(V.shape, device)
    with torch.no_grad():  # disables gradient calculation (reduces memory consumption - cannot use .backward() anymore)
        diff_Euler = V - V_alpha  # calculate deviation for agents to CP to (Nxd)-tensor-matrix for Euler step
        noise_weight_Euler = torch.abs(diff_Euler) if anisotropic else torch.norm(diff_Euler, p=2, dim=1).reshape(-1,1)  # calculate norm (1xN - since norm on each row) or absolute value (Nxd - bc componentwise multiplication of (Nxd) and (Nxd)) for Euler step
        V_Euler = - l * diff_Euler * dt + sigma * noise_weight_Euler * noise * (dt ** 0.5)  # calculate X^(star) - V
        batch_energy_values = compute_energy_values(function, V + V_Euler, device=device)  # function evaluation of all agents in the batch
        V_alpha_Heun = compute_v_alpha(batch_energy_values, V + V_Euler, alpha, device=device)  # compute CP for Heun step
        diff_Heun = (V + V_Euler) - V_alpha_Heun  # calculate deviation for agents to CP to (Nxd)-tensor-matrix for Heun step
        noise_weight_Heun = torch.abs(diff_Heun) if anisotropic else torch.norm(diff_Heun, p=2, dim=1)  # calculate norm (1xN - since norm on each row) or absolute value (Nxd - bc componentwise multiplication of (Nxd) and (Nxd))
        V += V_Euler + sigma/(torch.sqrt(torch.tensor([[2 * dt]]))) * (noise_weight_Heun - noise_weight_Euler) * ((noise * (dt ** 0.5)) ** 2 - dt)  # do explicit order 1.0 strong update
    return V










































# functions used by Heun
def compute_v_alpha(energy_values, agents, alpha, device=None):
    device = torch.device('cpu') if device is None else device
    weights = torch.exp(-alpha * (energy_values - energy_values.min())).reshape(-1, 1).to(device)  # turn vector in (batch_size x 1)-tensor-matrix
    consensus = (weights * agents) / weights.sum()  # calculate the weighted agents (differently weighted in each dimension)
    return consensus.sum(dim=0)  # sum all weighted agents together (in each dimension)

def compute_energy_values(function, agents, device=None):
    device = torch.device('cpu') if device is None else device
    return function(agents).to(device)  # function evaluation of each agent in V recorded in device