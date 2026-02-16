# This file contains the relevant functions for executing CBO


import torch
import numpy as np

from scipy.stats import qmc
from torch.utils.data import DataLoader
from ODE_methods import * # import all update methods (dependent on the SDE approximation scheme)


def compute_v_alpha(energy_values, agents, alpha, device=None):
    # INPUT: - energy_values ((1xbatch_size)-tensor-matrix): function evaluation of each agent in V 
    #        - agents ((batch_size x d)-tensor-matrix): set of d-dimensional agents/particles of the batch
    #        - alpha (scalar): agent/particle weight for consensu point
    #        - device (device): represents where torch.tensors will be allocated
    # OUTPUT: - V_alpha ((1xd)-tensor-matrix): consensus point of all agents 
    device = torch.device('cpu') if device is None else device
    weights = torch.exp(-alpha * (energy_values - energy_values.min())).reshape(-1, 1).to(device)  # turn vector in (batch_size x 1)-tensor-matrix
    # print(weights/weights.sum())
    consensus = (weights * agents) / weights.sum()  # calculate the weighted agents (differently weighted in each dimension)
    return consensus.sum(dim=0)  # sum all weighted agents together (in each dimension)


def compute_energy_values(function, agents, device=None):
    # compute function evaluation of all the particles/agents
    # INPUT: - function (function): optimization function f
    #        - agents ((batch_size x d)-tensor-matrix): set of d-dimensional agents/particles
    #        - device (device): represents where torch.tensors will be allocated
    # OUTPUT: - energy_values ((1xbatch_size)-tensor-matrix): function evaluation of each agent in V 
    device = torch.device('cpu') if device is None else device
    return function(agents).to(device)  # function evaluation of each agent in V recorded in device


def minimize(# INPUT:
        # General CBO / optimization parameters
        function, dimensionality, n_particles, K_rho, x_est, dt, l, alpha, SDE_method, theta,
        # function (function): optimization function f
        # dimensionality (integer): dimension of the optimization problem
        # n_particles (integer): number of total agents N = 2^n_particles
        # K_rho (scalar): radius of sample cube
        # a (scalar or D-dimensional numpy-array): center of the cube
        # dt (scalar): step size of discretization
        # l (scalar): drift parameter lambda
        # alpha (scalar): agent weight for CP 
        # SDE_method (string): determines the method to solve SDE from {'Euler','linear_Steklov','tamed_Euler','Theta','semi_Heun'}
        # theta (scalar): Implicity parameter for theta Method (only necessary if this Theta-method is used, theta = 0 is explicit Euler-Maruyama)
        # Optimization parameters
        batch_size=None, n_particles_batches=None, epochs=None, time_horizon=None,
        # batch_size (integer): number of agents per batch
        # n_particles_batches (integer): number of batches
        # epochs (integer): number of times the agents 'move'/change position 
        # time_horizon (scalar): defines time interval [0,time_horizon] we investigate the agent behaviour 
        # Optimization modifications parameters
        use_partial_update=False, use_additional_gradients_shift=False, gradients_shift_gamma=None,
        # use_partial_update (Boolean): if TRUE only batch agents change position, else all agents do 
        # use_additional_gradients_shift (Boolean): if TRUE use gradient shift after each epoch
        # gradients_shift_gamma (scalar): scale the indluence of the gradient 
        # Additional optional arguments
        best_particle_alpha=1e5, use_gpu_if_available=False, return_trajectory=False, cooling=False       
        # best_particle_alpha (scalar): alpha for estimating current best agent with the weighted average
        # use_gpu_if_available (Boolean): if TRUE use 'cuda' as device  
        # return_trajectory (Boolean): if TRUE return position of all agents, CP and current best agent after each epoch
        # cooling (Boolean): if TRUE use cooling strategy (steadily increasing alpha and slighty steadily decreasing sigma)
        ):
    # OUTPUT: - V_alpha ((1xd)-tensor-matrix): Prediction of CBO for the minimizer of the function 
    #         - trajectory (dictionary): stores all agent positions (epoch x N x d), all CP (epoch x 1 x d) and all best agents (epoch x 1 x d)

    # Setting up computations on GPU / CPU
    device = torch.device('cuda') if (use_gpu_if_available and torch.cuda.is_available()) else torch.device('cpu')
    # Standardize input arguments
    batch_size = int((2**n_particles) // n_particles_batches) if batch_size is None else batch_size
    epochs = int(time_horizon // dt) if epochs is None else epochs

    # Initialize particle positions
    sobol = qmc.Sobol(d=dimensionality, scramble=True, seed=42)
    points = sobol.random_base2(m=n_particles)
    V_np = x_est + 2 * K_rho * points - K_rho
    V = torch.from_numpy(V_np).to(dtype=torch.float32, device=device) # generate N quasi-Monte Carlo points of dimension D as initial stat x_0

    if use_additional_gradients_shift:
        V.requires_grad = True  # record all operations done on V
    V_batches = DataLoader(np.arange(2**n_particles), batch_size=batch_size, shuffle=True)

    trajectory = []
    # safe inital point distribution
    if return_trajectory:
        energy_values = compute_energy_values(function, V, device=device)  # compute function evaluation of each agent (1xN)-tensor-matrix
        V_alpha = compute_v_alpha(energy_values, V, alpha, device=device)  # compute current CP for all agents (1xd)-tensor-matrix
        V_best = compute_v_alpha(energy_values, V, best_particle_alpha, device=device)  # compute current best agent (with lowest function evaluation) via a very high alpha (convergence to inf on all agents)
        trajectory.append(
            {
                'V': V.clone().detach().cpu(),  # save positions of all agents (Nxd)-tensor-matrix
                'V_alpha': V_alpha.clone().detach().cpu(),  # save CP (1xd)-tensor-matrix
                'V_best': V_best.clone().detach().cpu(),  # save current best particle (1xd)-tensor-matrix
            }
        )

    # Main optimization loop
    for epoch in range(epochs):
        for batch in V_batches:
            V_batch = V[batch]
            batch_energy_values = compute_energy_values(function, V_batch, device=device)  # function evaluation of all agents in the batch
            V_alpha = compute_v_alpha(batch_energy_values, V_batch, alpha, device=device)  # compute consensus points

            if use_partial_update:
                # update agent positions of batch based on the SDE method
                if SDE_method == 'Euler':
                    V[batch] = update_Euler(V_batch, V_alpha, l, dt, device=device) 
                elif SDE_method == 'linear_Steklov':
                    V[batch] = update_linear_Steklov(V_batch, V_alpha, l, dt, function, alpha , device=device)
                elif SDE_method == 'tamed_Euler':
                    V[batch] = update_tamed_Euler(V_batch, V_alpha, l, dt, epoch, device=device)
                elif SDE_method == 'Theta':
                    V[batch] = update_Theta(V_batch, V_alpha, l, dt, theta, device=device)
                elif SDE_method == 'semi_Heun':
                    V[batch] = update_semi_Heun(V_batch, V_alpha, l, dt, function, alpha, device=device)
                else:
                    print('ERROR: The desired SDE solving method is NOT applicable')
                    return
            else:
                # update all agent positions based on the ODE method
                if SDE_method == 'Euler':
                    V = update_Euler(V, V_alpha, l, dt, device=device) 
                elif SDE_method == 'linear_Steklov':
                    V = update_linear_Steklov(V, V_alpha, l, dt, function, alpha , device=device)
                elif SDE_method == 'tamed_Euler':
                    V = update_tamed_Euler(V, V_alpha, l, dt, epoch, device=device)
                elif SDE_method == 'Theta':
                    V = update_Theta(V, V_alpha, l, dt, theta, device=device)
                elif SDE_method == 'semi_Heun':
                    V = update_semi_Heun(V, V_alpha, l, dt, function, alpha, device=device)
                else:
                    print('ERROR: The desired SDE solving method is NOT applicable')
                    return
                
        if use_additional_gradients_shift:
            if V.grad is not None:
                V.grad.zero_()  # set gradients to 0 before doing backpropagation
            energy_values = compute_energy_values(function, V, device=device)  # function evaluation of all agents (1xN-matrix)
            loss = energy_values.sum()  # sum all function evaluations
            loss.backward()
            with torch.no_grad():  # do not update gradient automatically so backpropagation can be done
                V -= gradients_shift_gamma * V.grad  # add gradient shift

        if return_trajectory:
            # save information for each epoch, if trajectory shall be returned
            energy_values = compute_energy_values(function, V, device=device)  # compute function evaluation of each agent (1xN)-tensor-matrix
            V_alpha = compute_v_alpha(energy_values, V, alpha, device=device)  # compute current CP for all agents (1xd)-tensor-matrix
            V_best = compute_v_alpha(energy_values, V, best_particle_alpha, device=device)  # compute current best agent (with lowest function evaluation) via a very high alpha (convergence to inf on all agents)
            trajectory.append(
                {
                    'V': V.clone().detach().cpu(),  # save positions of all agents (Nxd)-tensor-matrix
                    'V_alpha': V_alpha.clone().detach().cpu(),  # save CP (1xd)-tensor-matrix
                    'V_best': V_best.clone().detach().cpu(),  # save current best particle (1xd)-tensor-matrix
                }
            )

        if cooling:
            # use cooling strategy after each epoch, if wanted
            alpha = alpha * 2  # change alpha value
            sigma = sigma * np.log2(epoch + 1) / np.log2(epoch + 2)  # change sigma value

    energy_values = compute_energy_values(function, V, device=device)  # compute final function evaluation of each agent
    V_alpha = compute_v_alpha(energy_values, V, alpha, device=device)  # compute final CP
    if return_trajectory:
        return V_alpha.detach().cpu(), trajectory  # return approximation of min (final CP) and trajectory
    return V_alpha.detach().cpu()  # return approximation of min (final CP)


  