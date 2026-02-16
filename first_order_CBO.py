# This file contains the relevant functions for executing CBO


import torch
import numpy as np

from utils import inplace_randn
from torch.utils.data import DataLoader
from first_order_SDE_methods import * # import all update methods (dependent on the SDE approximation scheme)


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
        function, dimensionality, n_particles, initial_distribution, dt, l, sigma, alpha, anisotropic, SDE_method,
        # function (function): optimization function f
        # dimensionality (integer): dimension of the optimization problem
        # n_particles (integer): number of total agents N
        # initial_distribution (distribution): distribution to draw inital strate X_0 from
        # dt (scalar): step size of discretization
        # l (scalar): drift parameter lambda
        # sigma (scalar): diffusion parameter 
        # alpha (scalar): agent weight for CP 
        # anisotropic (Boolean): if TRUE use absolute difference instead of 2-norm in the stochastic part of the SDE 
        # SDE_method (string): determines the method to solve SDE from {'Euler_Maruyma','tamed_Euler','advanced_Heun','simple_Heun','order_1'}
        # Optimization parameters
        batch_size=None, n_particles_batches=None, epochs=None, time_horizon=None,
        # batch_size (integer): number of agents per batch
        # n_particles_batches (integer): number of batches
        # epochs (integer): number of times the agents 'move'/change position 
        # time_horizon (scalar): defines time interval [0,time_horizon] we investigate the agent behaviour 
        # Optimization modifications parameters
        use_partial_update=False, use_additional_random_shift=False, use_additional_gradients_shift=False, random_shift_epsilon=None, gradients_shift_gamma=None,
        # use_partial_update (Boolean): if TRUE only batch agents change position, else all agents do 
        # use_additional_random_shift (Boolean): if TRUE add additional random shift to agents when change of CP ist low 
        # use_additional_gradients_shift (Boolean): if TRUE use gradient shift after each epoch
        # random_shift_epsilon (scalar): determines how low change of CP has to be to add additional random noise
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
    batch_size = int(n_particles // n_particles_batches) if batch_size is None else batch_size
    epochs = int(time_horizon // dt) if epochs is None else epochs
    # Initialize variables
    V = initial_distribution.sample((n_particles, dimensionality)).to(device)  # generate RN of size Nxd as initial state X_0
    if use_additional_gradients_shift:
        V.requires_grad = True  # record all operations done on V
    V_batches = DataLoader(np.arange(n_particles), batch_size=batch_size, shuffle=True)  # gives batches of size batch_size of all N agents and shuffles them (stores the indices of all for all batches)
    V_alpha_old = None

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
            # print(V_alpha)

            if use_partial_update:
                # update agent positions of batch based on the SDE method
                if SDE_method == 'Euler_Maruyama':
                    V[batch] = first_order_cbo_update_Euler_Maruyama(V_batch, V_alpha, anisotropic, l, sigma, dt, device=device)  
                elif SDE_method == 'tamed_Euler':
                    V[batch] = first_order_cbo_update_tamed_Euler(V_batch, V_alpha, anisotropic, l, sigma, dt, epoch, device=None)
                elif SDE_method == 'advanced_Heun':
                    V[batch] = first_order_cbo_update_advanced_Heun(V_batch, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=device) 
                elif SDE_method == 'simple_Heun':
                    V[batch] = first_order_cbo_update_simple_Heun(V_batch, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=device) 
                elif SDE_method == 'order_1':
                    V[batch] = first_order_cbo_update_order_1(V_batch, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=device)
                else:
                    print('ERROR: The desired SDE solving method is NOT applicable')
                    return
            else:
                # update all agent positions based on the SDE method
                if SDE_method == 'Euler_Maruyama':
                    V = first_order_cbo_update_Euler_Maruyama(V, V_alpha, anisotropic, l, sigma, dt, device=device)  
                elif SDE_method == 'tamed_Euler':
                    V = first_order_cbo_update_tamed_Euler(V, V_alpha, anisotropic, l, sigma, dt, epoch, device=None)
                elif SDE_method == 'advanced_Heun':
                    V = first_order_cbo_update_advanced_Heun(V, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=device) 
                elif SDE_method == 'simple_Heun':
                    V = first_order_cbo_update_simple_Heun(V, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=device) 
                elif SDE_method == 'order_1':
                    V = first_order_cbo_update_order_1(V, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=device)
                else:
                    print('ERROR: The desired SDE solving method is NOT applicable')
                    return

            if use_additional_random_shift:
                if V_alpha_old is None:
                    V_alpha_old = V_alpha  # update V_alpha_old
                    continue  # Don't do anything in first iteration
                norm = torch.norm(V_alpha.view(-1) - V_alpha_old.view(-1), p=float('inf'), dim=0).detach().cpu().numpy()  # calculate maximal deviation of old CP and new CP (in terms of all dimensions)
                if np.less(norm, random_shift_epsilon):  # add random shift if norm < random_shift_epsilon (change of CP was very small in all dimension)
                    V += sigma * (dt ** 0.5) * inplace_randn(V.shape, device=device) # add random shift
                V_alpha_old = V_alpha  # update old alpha

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





# this function does the cbo procedure but we will start with a given set of particles instead of random starting distributions
def minimize_with_starting(# INPUT:
        # General CBO / optimization parameters
        function, dimensionality, n_particles, initial_points, dt, l, sigma, alpha, anisotropic, SDE_method,
        # initial_points ((Nxd)-tensor-matrix): starting points for CBO
        batch_size=None, n_particles_batches=None, epochs=None, time_horizon=None,
        use_partial_update=False, use_additional_random_shift=False, use_additional_gradients_shift=False, random_shift_epsilon=None, gradients_shift_gamma=None,
        best_particle_alpha=1e5, use_gpu_if_available=False, return_trajectory=False, cooling=False       
        ):

    device = torch.device('cuda') if (use_gpu_if_available and torch.cuda.is_available()) else torch.device('cpu')
    batch_size = int(n_particles // n_particles_batches) if batch_size is None else batch_size
    epochs = int(time_horizon // dt) if epochs is None else epochs
    # Initialize variables
    V = initial_points # generate RN of size Nxd as initial state X_0
    if use_additional_gradients_shift:
        V.requires_grad = True  # record all operations done on V
    V_batches = DataLoader(np.arange(n_particles), batch_size=batch_size, shuffle=True)  # gives batches of size batch_size of all N agents and shuffles them (stores the indices of all for all batches)
    V_alpha_old = None   
    trajectory = [] 
    if return_trajectory:
        energy_values = compute_energy_values(function, V, device=device)  # function evaluation of all agents in the batch
        V_best = compute_v_alpha(energy_values, V, best_particle_alpha, device=device)  # compute current best agent (with lowest function evaluation) via a very high alpha (convergence to inf on all agents)
        V_alpha = compute_v_alpha(energy_values, V, alpha, device=device)  # compute consensus points
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
            # print(V_alpha)

            if use_partial_update:
                # update agent positions of batch based on the SDE method
                if SDE_method == 'Euler_Maruyama':
                    V[batch] = cbo_update_Euler_Maruyama(V_batch, V_alpha, anisotropic, l, sigma, dt, device=device)  
                elif SDE_method == 'tamed_Euler':
                    V[batch] = cbo_update_tamed_Euler(V_batch, V_alpha, anisotropic, l, sigma, dt, epoch, device=None)
                elif SDE_method == 'advanced_Heun':
                    V[batch] = cbo_update_advanced_Heun(V_batch, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=device) 
                elif SDE_method == 'simple_Heun':
                    V[batch] = cbo_update_simple_Heun(V_batch, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=device) 
                elif SDE_method == 'order_1':
                    V[batch] = cbo_update_order_1(V_batch, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=device)
                else:
                    print('ERROR: The desired SDE solving method is NOT applicable')
                    return
            else:
                # update all agent positions based on the SDE method
                if SDE_method == 'Euler_Maruyama':
                    V = cbo_update_Euler_Maruyama(V, V_alpha, anisotropic, l, sigma, dt, device=device)  
                elif SDE_method == 'tamed_Euler':
                    V = cbo_update_tamed_Euler(V, V_alpha, anisotropic, l, sigma, dt, epoch, device=None)
                elif SDE_method == 'advanced_Heun':
                    V = cbo_update_advanced_Heun(V, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=device) 
                elif SDE_method == 'simple_Heun':
                    V = cbo_update_simple_Heun(V, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=device) 
                elif SDE_method == 'order_1':
                    V = cbo_update_order_1(V, V_alpha, anisotropic, l, sigma, dt, function, alpha, device=device)
                else:
                    print('ERROR: The desired SDE solving method is NOT applicable')
                    return

            if use_additional_random_shift:
                if V_alpha_old is None:
                    V_alpha_old = V_alpha  # update V_alpha_old
                    continue  # Don't do anything in first iteration
                norm = torch.norm(V_alpha.view(-1) - V_alpha_old.view(-1), p=float('inf'), dim=0).detach().cpu().numpy()  # calculate maximal deviation of old CP and new CP (in terms of all dimensions)
                if np.less(norm, random_shift_epsilon):  # add random shift if norm < random_shift_epsilon (change of CP was very small in all dimension)
                    V += sigma * (dt ** 0.5) * inplace_randn(V.shape, device=device) # add random shift
                V_alpha_old = V_alpha  # update old alpha

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

