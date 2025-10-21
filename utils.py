import os
import time
import numpy as np
import json
import pandas as pd
from scipy.linalg import orth
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatter
from scipy.optimize import differential_evolution, NonlinearConstraint
from tqdm import tqdm
from sklearn import preprocessing

np.random.seed(42)

def run_alt_gd_min_ada(epoch_source_data, 
                   epoch_source_labels, 
                   target_data, 
                   target_labels, 
                   order_tasks, 
                   params, 
                   Theta_star=None):
    d = params["d"]
    k = params["k"]
    M  = params["M"]
    M_global = M
    epochs = params["epochs"]
    C = params["C"]
    c = params["c"]
    gd_iterations = params["gd_iterations"]
    num_source_sample = params["num_source_sample"]
    increase_gd_iteration = params["increase_gd_iteration"]
    samples_per_epoch = params["samples_per_task"]  
    num_target_sample = params["num_target_sample"]
    target_corruption = params["target_corruption"]
    target_label = params["target_label"]
    cumulative_time = 0
    nu = generate_nu(M)
    nu_hat = [1 / M for m in range(M)]
    distribution = []
    N_i = num_source_sample  
    N = N_i * epochs
    config_file = "config/AltGD_ADA_" +target_corruption +"_"+str(target_label)+"_numtasks_"+str(M)+ "_k_" +str(k) + ".json"
    ER_list1  = []
    ER_1, gd_time_1, epoch_time_1 = [], [], []
    nu_error_list = []
    error_list_1 = []
    percentage_error = []


    for i in range(epochs):
        
        print("Running Epoch:", i)
        start_time = time.time()
        source_data, source_label = epoch_source_data[i], epoch_source_labels[i]
        X, Y = [], []
        index_m = []

        for m in range(M):
            if i == 0:
                n_m = samples_per_epoch  
            else:
                nu_squared_sum = sum([nu_hat[j]**2 for j in range(M)])
                if nu_squared_sum > 0:
                    n_m = int(num_source_sample * (nu_hat[m]**2) / nu_squared_sum)
                else:
                    n_m = samples_per_epoch  
            
            max_available_samples = source_data[m].shape[0]
            n_m = min(n_m, max_available_samples)
                
            if i == 3:
                distribution.append(n_m)

            if n_m == 0:
                
                M = M - 1
                index_m.append(m)
                
                continue
        
            X_temp = np.zeros((d, n_m))
            Y_temp = np.zeros(n_m)
            
            for num in range(n_m):
                X_temp[:, num] = source_data[m][num]
                Y_temp[num] = source_label[m][num]

            X.append(X_temp)
            Y.append(Y_temp)

        if len(index_m) > 0:
            
            W_hat_m = W_hat[:, index_m]

        if i == 0:
            B_hat, W_hat, gd_time_1, error_list = AltGD_Min_1(X, Y, c, C, M, d, k, gd_iterations, gd_time_1, Theta_star) 
            error_list_1.append(error_list)
            iterations = range(len(error_list_1[0]))
            plt.plot(iterations, error_list_1[0])
            plt.show()
        else:
            
            B_hat, W_hat = AltGD_Min_2(X, Y, c, M, d, k, B_hat, gd_iterations + (i - 1) * increase_gd_iteration)

        if len(index_m) > 0:
            
            j = 0
            W_temp = np.zeros((k, M + len(index_m)))
            
            for v in range(M + len(index_m)):
                
                if v in index_m:
                    
                    continue
                    
                W_temp[:, v] = W_hat[:, j]
                j += 1
                
            for idx, column in zip(index_m, W_hat_m.T):
                
                W_temp[:, idx] = column
                
            W_hat = W_temp
            M = M_global
            
        Theta_hat = B_hat.dot(W_hat)

        error = np.linalg.norm(Theta_star - Theta_hat, 'fro') / np.linalg.norm(Theta_star, 'fro')
        print(f'k: {k} epoch: {i} Theta reconstruction error: {error:.6f}')

        X_target = np.zeros((num_target_sample, d))
        Y_target = np.zeros(num_target_sample)

        for num in range(num_target_sample):
            X_target[num, :] = target_data[num]
            Y_target[num] = target_labels[num] 

        w_hat_target = np.linalg.pinv(X_target.dot(B_hat)).dot(Y_target)
        
       
        nu_hat = W_hat.T.dot(np.linalg.pinv(np.dot(W_hat, W_hat.T))).dot(w_hat_target)
        # print("Estimated nu_hat:", nu_hat)
        # rank_w_hat = np.linalg.matrix_rank(W_hat)
        # print("Rank of W_hat:", rank_w_hat)

        end_time = time.time()
        term1 = X_target.dot(B_hat).dot(w_hat_target)
        theta_star_target = B_hat.dot(w_hat_target)
        ER_target_task = np.sum((theta_star_target - Theta_star[:, -1]) ** 2) / np.linalg.norm(Theta_star[:, -1])**2
        # print("Construction error for target task:", ER_target_task)
        ER = np.sum((term1 - Y_target) ** 2) / num_target_sample

        cumulative_time = cumulative_time + end_time - start_time
        ER_1.append(ER)
        epoch_time_1.append(cumulative_time)

        if i == 3:
            
            print(f'k: {k} epoch: {i} ER: {ER}')

    mapping = {}
    
    for i, task in enumerate(order_tasks):
        
        mapping[task] = distribution[i]
    
    
    ER_list1.append((k, ER_1))
    params['ER'] = ER_list1
    # params['percentage_error'] = error
    # params['nu_error'] = nu_error_list
    params['error_list'] = error_list_1
    params['nu_hat'] = mapping

    # Plot error_list_1 for AltGD_ADA

    with open(config_file, 'w') as f:
        
        json.dump(params, f, indent=4)


def run_alt_gd_min_noada( epoch_source_data, 
                        epoch_source_labels, 
                        target_data, 
                        target_labels, 
                        order_tasks, 
                        params, 
                       Theta_star):
    d = params["d"]
    k = params["k"]
    M = params["M"]
    M_global = M
    epochs = params["epochs"]
    C = params["C"]
    c = params["c"]
    gd_iterations = params["gd_iterations"]
    num_source_sample = params["num_source_sample"]
    increase_gd_iteration = params["increase_gd_iteration"]
    samples_per_epoch = params["samples_per_task"]  # Use samples_per_task for uniform
    num_target_sample = params["num_target_sample"]
    target_corruption = params["target_corruption"]
    target_label = params["target_label"]
    config_file = "config/AltGD_NOADA_" +target_corruption +"_"+str(target_label)+"_numtasks_"+str(M) +  "_k_" +str(k) +".json"

    cumulative_time = 0
    N_i = num_source_sample  # Total samples per epoch
    N = N_i * epochs

    ER_list1, percentage_error_list1 = [], [], 
    ER_1, gd_time_1, epoch_time_1 = [], [], []
    
    # print("Source Data Shape:",epoch_source_data.shape)
    # print("Target Data Shape:",target_data.shape)
    Theta_star = Theta_star
    nu_error_list = []
    percentage_error = []
    error_list_1 = []

    for i in range(epochs):
        
        start_time = time.time()
        source_data, source_label = epoch_source_data[i], epoch_source_labels[i]
        
        X, Y = [], []

        for m in range(M):
            
            # AltGD_NoADA uses uniform sampling: each task gets the same number of samples every epoch
            n_m = samples_per_epoch  # This equals samples_per_task for uniform distribution
            
            X_temp = np.zeros((d, n_m))
            Y_temp = np.zeros(n_m)
            
            for num in range(n_m):
                
                X_temp[:, num] = source_data[m][num]
                Y_temp[num] = source_label[m][num]  # Removed noise

            X.append(X_temp)
            Y.append(Y_temp)

        if i == 0:
            
            B_hat, W_hat, gd_time_1, erlist = AltGD_Min_1(X, Y, c, C, M, d, k, gd_iterations, gd_time_1, Theta_star) 
            error_list_1.append(erlist)
            
        else:
            
            B_hat, W_hat = AltGD_Min_2(X, Y, c, M, d, k, B_hat, gd_iterations + (i - 1) * increase_gd_iteration)

        X_target = np.zeros((num_target_sample, d))
        Y_target = np.zeros(num_target_sample)

        for num in range(num_target_sample):
            
            X_target[num, :] = target_data[num]
            Y_target[num] = target_labels[num]

        Theta_hat = np.dot(B_hat, W_hat)
        error = np.linalg.norm(Theta_star - Theta_hat, 'fro') / np.linalg.norm(Theta_star, 'fro')

        w_hat_target = np.linalg.pinv(X_target.dot(B_hat)).dot(Y_target)
        nu_hat = W_hat.T.dot(np.linalg.pinv(np.dot(W_hat, W_hat.T))).dot(w_hat_target)

        end_time = time.time()
        term1 = X_target.dot(B_hat).dot(w_hat_target)
        ER = np.sum((term1 - Y_target) ** 2) / num_target_sample

        cumulative_time = cumulative_time + end_time - start_time
        ER_1.append(ER)
        percentage_error.append(error)
        # nu_error_list.append(nu_error)
        epoch_time_1.append(cumulative_time)
        # print(f'k: {k} epoch: {i} ER: {ER}')
        
        if i == 3:
            
            print(f'k: {k} epoch: {i} ER: {ER}')
    
    ER_list1.append((k, ER_1))
    params['ER'] = ER_list1
    params['error_list'] = error_list_1
    params['percentage_error'] = percentage_error
    params['nu_error_list'] = nu_error_list
    params['percentage_error_list1'] = percentage_error_list1
    params['gd_time_list1'] = gd_time_1
    params['epoch_time_list1'] = epoch_time_1

    with open(config_file, 'w') as f:
        
        json.dump(params, f, indent=4)


def run_chen(epoch_source_data, 
            epoch_source_labels, 
            target_data, 
            target_labels, 
            order_tasks, 
            params, 
            Theta_star=None):
    d = params["d"]
    k = params["k"]
    M  = params["M"]
    epochs = params["epochs"]
    gd_iterations_chen = params["gd_iterations"]
    num_source_sample = params["num_source_sample"]
    samples_per_epoch = params["samples_per_task"]  # Use samples_per_task
    num_target_sample = params["num_target_sample"]
    learning_rate = params["learning_rate"]
    target_corruption = params["target_corruption"]
    target_label = params["target_label"]
    config_file = "config/Chen_ADA_" + target_corruption + "_"+str(target_label) + "_numtasks_" + str(M) +  "_k_" +str(k) + ".json"

    cumulative_time = 0
    nu_hat = [1 / M for m in range(M)]
    N_i = num_source_sample  # Total samples per epoch
    N = N_i * epochs

    # print("Epoch Source Sample Images:", epoch_source_data.shape)
    # print("Epoch Source Sample Labels:", epoch_source_labels.shape)
    # print("Target Sample Images:", target_data.shape)
    # print("Target Sample Labels:", target_labels.shape)
    
    # Chen et al. Algorithm
    start = time.time()
    ER_3 = []
    epoch_time_3 = []
    cumulative_time = 0

    distribution = []
    percentage_error = []
    nu_error_list = []
    error_list_1 = []

    for i in range(epochs):
        
        print("Running Epoch:", i)
        start_time = time.time()
        
        source_data, source_label = epoch_source_data[i], epoch_source_labels[i]

        X = []
        Y = []
        
        stable_learning_rate = min(learning_rate, 1e-6)

        cnt = 0 
        temp = np.abs(nu_hat) / np.linalg.norm(nu_hat, 1)
        for ele in temp:
            if abs(ele - 0) > 0.0001:
                cnt += 1
                
        for m in range(M):
            # Chen et al. uses adaptive sampling with constant total budget
            n_m = int(num_source_sample * (np.abs(nu_hat) / np.linalg.norm(nu_hat, 1))[m])
            
            # Ensure n_m doesn't exceed available samples for this task
            max_available_samples = source_data[m].shape[0]
            n_m = min(n_m, max_available_samples)
            
            if i == 3:
                
                distribution.append(n_m)
                
            X_temp = np.zeros((d, n_m))
            Y_temp = np.zeros(n_m)
            
            for num in range(n_m):
                X_temp[:, num] = source_data[m][num]
                Y_temp[num] = source_label[m][num]  # Removed noise

            X.append(X_temp)
            Y.append(Y_temp)
            
        Theta_hat, error_list = gd_Theta(X, Y, M, d, gd_iterations_chen, stable_learning_rate, Theta_star)
        error_list_1.append(error_list)
        
        Theta_reg = Theta_hat + 1e-8 * np.eye(Theta_hat.shape[0], Theta_hat.shape[1])
        B_hat, Sigma_hat, V_hat = np.linalg.svd(Theta_reg, full_matrices=False)
        B_hat = B_hat[:, :k]
        W_hat = np.diag(Sigma_hat[:k]).dot(V_hat[:k, :])
     
        
        X_target = np.zeros((num_target_sample, d))
        Y_target = np.zeros(num_target_sample)
        
        for num in range(num_target_sample):
            
            X_target[num, :] = target_data[num]
            # Fix: Don't add noise to target labels since noise array doesn't include target
            Y_target[num] = target_labels[num]  # Remove noise for now
            
        w_hat_target = np.linalg.pinv(X_target.dot(B_hat)).dot(Y_target)
        error = np.linalg.norm(Theta_star - Theta_hat, 'fro') / np.linalg.norm(Theta_star, 'fro')
        def objective(nu_hat):
            
            return np.linalg.norm(nu_hat)
        
        constraint = {'type': 'eq', 'fun' : lambda nu_hat: np.sum(np.dot(W_hat, nu_hat) - w_hat_target)}
        
        optimal_nu_hat = minimize(
            fun = objective,
            x0 = nu_hat,
            method='SLSQP',
            constraints= [constraint]
        )
        
        nu_hat = optimal_nu_hat.x
        nu_error = 0 
            
        end_time = time.time()
        
        term1 = X_target.dot(B_hat).dot(w_hat_target)
        theta_star_target = B_hat.dot(w_hat_target)
        ER_target_task = np.sum((theta_star_target - Theta_star[:, -1]) ** 2) / np.linalg.norm(Theta_star[:, -1])**2
        print("Construction error for target task:", ER_target_task)
        ER = np.sum((term1 - Y_target) ** 2) / num_target_sample
        
        cumulative_time = cumulative_time + end_time - start_time
        
        ER_3.append(ER)
        percentage_error.append(error)
        nu_error_list.append(nu_error)
        epoch_time_3.append(cumulative_time)
        
        
        if i == 3:
            
            print(f'k: {k} epoch: {i} ER: {ER}')

    mapping = {}
    
    for i, task in enumerate(order_tasks):
        
        mapping[task] = distribution[i]
    params['ER'] = ER_3
    params['nu_hat'] = mapping
    params['percentage_error'] = percentage_error
    params['nu_error_list'] = nu_error_list
    params['epoch_time_3'] = epoch_time_3
    end = time.time()
    print('Finished! The total time we use is: ', end - start)

    with open(config_file, 'w') as f:
        
        json.dump(params, f, indent=4)


def run_mom(epoch_source_data, 
            epoch_source_labels, 
            target_data, 
            target_labels, 
            order_tasks, 
            params, 
            Theta_star=None):
    d = params["d"]
    k = params["k"]
    M = params["M"]
    epochs = params["epochs"]
    samples_per_task = params["samples_per_task"]  
    num_target_sample = params["num_target_sample"]
    target_corruption = params["target_corruption"]
    target_label = params["target_label"]
    config_file = "config/MOM_"+target_corruption+"_"+str(target_label)+"_numtasks_"+ str(M) + "_k_" +str(k)+".json"
    ER_2 = []
    epoch_time_2 = []

    cumulative_time = 0
    N_i = params["num_source_sample"] 
    N = N_i * epochs
    ld_0 = 1

    B_hat = np.zeros((d, k))
    W_hat = np.zeros((k, M))
    Theta_hat = B_hat.dot(W_hat)
    
    percentage_error = []
    nu_error_list = []
    error_list_1 = []
    M_hat = np.zeros((d, d))
    
    nu_hat = [1 / M for m in range(M)]
    distribution = []
    
    for i in range(epochs):
            
        start_time = time.time()
        source_data, source_label = epoch_source_data[i], epoch_source_labels[i]
    
        if i == 0:
            
            n_m_i = 0
            
            for m in range(M):
                n_m = int((N / epochs) * (np.abs(nu_hat) / np.linalg.norm(nu_hat, 1))[m])
                n_m_i += n_m
                
                for num in range(n_m):
                    
                    x_temp = source_data[m][num]
                    y_temp = source_label[m][num]  # No noise
                    M_hat = M_hat + y_temp ** 2 * np.dot(x_temp.T, x_temp)
                    
            M_hat = (1 / n_m_i) * M_hat
            
            try:
                M_reg = M_hat + 1e-8 * np.eye(M_hat.shape[0])
                U, Sigma, V = np.linalg.svd(M_reg, full_matrices=False)
                B_hat = U[:, :k]
            except np.linalg.LinAlgError:
                print("SVD failed in MOM, using random initialization")
                B_hat = np.random.randn(d, k) * 0.1
        
            X_target = np.zeros((num_target_sample, d))
            Y_target = np.zeros(num_target_sample)
            
            for num in range(num_target_sample):
                
                X_target[num, :] = target_data[num]
                Y_target[num] = target_labels[num]  # No noise
                    
                
            w_hat_target = np.linalg.pinv(X_target.dot(B_hat)).dot(Y_target)
            
            end_time = time.time()
            
            term1 = X_target.dot(B_hat).dot(w_hat_target)
            ER = np.sum((term1 - Y_target) ** 2) / num_target_sample
            
            cumulative_time = cumulative_time + end_time - start_time
            
            ER_2.append(ER)
            
        elif i == 1:
            
            X = []
            Y = []
            
            for m in range(M):
                
                # MOM uses adaptive sampling
                n_m = int((N / epochs) * (np.abs(nu_hat) / np.linalg.norm(nu_hat, 1))[m])
                X_temp = np.zeros((d, int(n_m / k) * k))
                Y_temp = np.zeros(int(n_m / k) * k)
                
                for num in range(int(n_m / k)):
                    
                    for k_value in range(k):
                        
                        X_temp[:, num * k + k_value] = np.sqrt(ld_0) * B_hat[:, k_value]
                        Y_temp[num * k + k_value] = source_label[m][num]  # No noise
                        
                X.append(X_temp)
                Y.append(Y_temp)
                
            for m in range(M):
                
                temp = X[m].T.dot(B_hat)
                W_hat[:, m] = np.linalg.lstsq(temp, Y[m], rcond=None)[0]
                
            Theta_hat = B_hat.dot(W_hat)
            error = np.linalg.norm(Theta_star - Theta_hat, 'fro') / np.linalg.norm(Theta_star, 'fro')
            
            X_target = np.zeros((num_target_sample, d))
            Y_target = np.zeros(num_target_sample)
            
            for num in range(num_target_sample):
                
                X_target[num, :] = target_data[num]
                Y_target[num] = target_labels[num]  # No noise
                
            w_hat_target = np.linalg.pinv(X_target.dot(B_hat)).dot(Y_target)
            
            nu_hat = W_hat.T.dot(np.linalg.pinv(np.dot(W_hat, W_hat.T))).dot(w_hat_target)
            # No ground truth available, so set nu_error to 0
            nu_error = 0
            
            end_time = time.time()
            
            term1 = X_target.dot(B_hat).dot(w_hat_target)
            ER = np.sum((term1 - Y_target) ** 2) / num_target_sample
            
            cumulative_time = cumulative_time + end_time - start_time
            
            ER_2.append(ER)
            nu_error_list.append(nu_error)
            percentage_error.append(error)
            epoch_time_2.append(cumulative_time)
            
            print('k: {} epoch: {} ER: {}:'.format(k, i, ER))
            
        else:
            
            for m in range(M):
                n_m = int((N / epochs) * (np.abs(nu_hat) / np.linalg.norm(nu_hat, 1))[m])
                if i == 3:
                    distribution.append(n_m)
            
            X_target = np.zeros((num_target_sample, d))
            Y_target = np.zeros(num_target_sample)
            
            for num in range(num_target_sample):
                
                X_target[num, :] = target_data[num]
                Y_target[num] = target_labels[num]  # No noise
                
            w_hat_target = np.linalg.pinv(X_target.dot(B_hat)).dot(Y_target)
            
            nu_hat = W_hat.T.dot(np.linalg.pinv(np.dot(W_hat, W_hat.T))).dot(w_hat_target)
            # No ground truth available, so set nu_error to 0
            nu_error = 0
            
            end_time = time.time()
            
            term1 = X_target.dot(B_hat).dot(w_hat_target)
            ER = np.sum((term1 - Y_target) ** 2) / num_target_sample
            
            cumulative_time = cumulative_time + end_time - start_time
            
            ER_2.append(ER)
            nu_error_list.append(nu_error)
            epoch_time_2.append(cumulative_time)
            
            if i == 3:
                
                print(f'k: {k} epoch: {i} ER: {ER}')
            
        
    mapping = {}
    
    for i, task in enumerate(order_tasks):
        
        mapping[task] = distribution[i] if i < len(distribution) else 0
        
    params['ER'] = ER_2
    params['percentage_error'] = percentage_error
    params['nu_error_list'] = nu_error_list
    params['nu_hat'] = mapping
    
    with open(config_file, 'w') as f:
        
        json.dump(params, f, indent=4)



def run_Collins(epoch_source_data, 
                epoch_source_labels, 
                target_data, 
                target_labels, 
                order_tasks, 
                params):
    d = params["d"]
    k = params["k"]
    M  = params["M"]
    M_global = M
    epochs = params["epochs"]
    C = params["C"]
    c = params["c"]
    gd_iterations = params["gd_iterations"]
    num_source_sample = params["num_source_sample"]
    increase_gd_iteration = params["increase_gd_iteration"]
    samples_per_epoch = params["samples_per_task"]  # Use samples_per_task for uniform
    num_target_sample = params["num_target_sample"]
    target_corruption = params["target_corruption"]
    target_label = params["target_label"]
    cumulative_time = 0
    nu = generate_nu(M)
    nu_hat = [1 / M for m in range(M)]
    distribution = []
    N_i = params["num_source_sample"]  # Total samples per epoch
    N = N_i * epochs
    config_file = "config/Collins_" +target_corruption +"_"+str(target_label)+"_numtasks_"+str(M) +  "_k_" +str(k) +".json"
    ER_list1, percentage_error_list1 = [], []
    ER_1, gd_time_1, epoch_time_1 = [], [], []
    percentage_error = []
    nu_error_list = []
    error_list_1 = []

    for i in range(epochs):
        
        print("Running Epoch:", i)
        
        start_time = time.time()
        source_data, source_label = epoch_source_data[i], epoch_source_labels[i]
        
        X, Y = [], []
        index_m = []

        for m in range(M):
            
            # Collins et al. uses uniform sampling (baseline)
            n_m = samples_per_epoch
            
            if i == 3:
                
                distribution.append(n_m)

            X_temp = np.zeros((d, n_m))
            Y_temp = np.zeros(n_m)
            for num in range(n_m):
                X_temp[:, num] = source_data[m][num]
                Y_temp[num] = source_label[m][num]  # Removed noise

            X.append(X_temp)
            Y.append(Y_temp)

        if len(index_m) > 0:
            
            W_hat_m = W_hat[:, index_m]

        if i == 0:
            U_init, Sigma_max = init_altgdmin_1(X, Y, C, M, d, k)
            B_hat = np.copy(U_init)
            W_hat = np.zeros((k, M))
            
            for m in range(M):
                W_hat[:, m] = np.linalg.pinv(np.dot(X[m].T, B_hat)).dot(Y[m])
            
        else:
            
            B_hat, W_hat = AltGD_Min_Collins2(X, Y, M, d, k, B_hat, gd_iterations + (i - 1) * increase_gd_iteration)

        if len(index_m) > 0:
            
            j = 0
            W_temp = np.zeros((k, M + len(index_m)))
            
            for v in range(M + len(index_m)):
                
                if v in index_m:
                    
                    continue
                    
                W_temp[:, v] = W_hat[:, j]
                j += 1
                
            for idx, column in zip(index_m, W_hat_m.T):
                
                W_temp[:, idx] = column
                
            W_hat = W_temp
            M = M_global
            
        Theta_hat = B_hat.dot(W_hat)
        # error = np.linalg.norm(Theta_star - Theta_hat, 'fro') / np.linalg.norm(Theta_star, 'fro')
        error = 0  # Set to 0 since Theta_star not available
        
        X_target = np.zeros((num_target_sample, d))
        Y_target = np.zeros(num_target_sample)

        for num in range(num_target_sample):
            
            X_target[num, :] = target_data[num]
            Y_target[num] = target_labels[num]

        w_hat_target = np.linalg.pinv(X_target.dot(B_hat)).dot(Y_target)
        nu_hat = W_hat.T.dot(np.linalg.pinv(np.dot(W_hat, W_hat.T))).dot(w_hat_target)
        # nu_error = np.linalg.norm(nu_hat - np.linalg.pinv(W_star).dot(w_target_star)) / np.linalg.norm(np.linalg.pinv(W_star).dot(w_target_star))
        nu_error = 0  # Set to 0 since W_star and w_target_star not available

        end_time = time.time()
        
        term1 = X_target.dot(B_hat).dot(w_hat_target)
        ER = np.sum((term1 - Y_target) ** 2) / num_target_sample

        cumulative_time = cumulative_time + end_time - start_time
        
        ER_1.append(ER)
        percentage_error.append(error)
        nu_error_list.append(nu_error)
        epoch_time_1.append(cumulative_time)
        
        if i == 3:
            
            print(f'k: {k} epoch: {i} ER: {ER}')


    mapping = {}
    
    for i, task in enumerate(order_tasks):
        
        mapping[task] = distribution[i]
        
    ER_list1.append((k, ER_1))
    params['ER'] = ER_list1
    params['percentage_error'] = percentage_error
    params['nu_error_list'] = nu_error_list
    params['error_list_1'] = error_list_1
    params['epoch_time_list1'] = epoch_time_1
    params['nu_hat'] = mapping

    with open(config_file, 'w') as f:
        
        json.dump(params, f, indent=4)


def data_loader_related_tasks(target_corruption, target_label, num_random_task, num_target_sample, num_source_sample, all_data_load, M):
    data_path = "data/mnist_c_processed"
    corruptions = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and not d.startswith('.')]

    task_to_index = {}
    current_task = 0
    for corrupt in corruptions:
        corruption_label_path= os.path.join(data_path,corrupt) 
        corruption_label = [d for d in os.listdir(corruption_label_path) if os.path.isdir(os.path.join(corruption_label_path, d)) and not d.startswith('.')]
        for task in corruption_label:
            task_to_index[task] = current_task
            current_task += 1

    random_tasks = []
    for coor in corruptions:
        for label in range(10):
            if coor != target_corruption and label != target_label:
                random_tasks.append(coor + "*" + str(label))
    order_tasks = []
    
    all_image_data = []
    all_label_data = []

    # Get all task within the same corruption 
    for i in range(10):
        if i != target_label:
            image_file_path = os.path.join(data_path,target_corruption, target_corruption +"_"+str(i), "train_images.npy") 
            label_file_path = os.path.join(data_path,target_corruption, target_corruption +"_"+str(i), "train_labels.npy") 
            order_tasks.append(target_corruption +"_"+str(i))
            train_images = np.load(image_file_path)
            train_labels = np.load(label_file_path)
            train_images = train_images/255
            strain_images = train_images[:all_data_load]
            train_labels = train_labels[:all_data_load]
            strain_images= strain_images.reshape(strain_images.shape[0], -1)
            strain_images = preprocessing.normalize(strain_images)
            all_image_data.append(strain_images)
            all_label_data.append(train_labels)
    
    # Get all task within the same label
    for i,corrup in enumerate(corruptions):
        if corrup != target_corruption:
            image_file_path = os.path.join(data_path,corrup, corrup +"_"+str(target_label), "train_images.npy") 
            label_file_path = os.path.join(data_path,corrup, corrup +"_"+str(target_label), "train_labels.npy") 
            order_tasks.append(corrup +"_"+str(target_label))
            train_images = np.load(image_file_path)
            train_labels = np.load(label_file_path)
            train_images = train_images/255
            strain_images = train_images[:all_data_load]
            train_labels = train_labels[:all_data_load]
            strain_images= strain_images.reshape(strain_images.shape[0], -1)
            strain_images = preprocessing.normalize(strain_images)
            all_image_data.append(strain_images)
            all_label_data.append(train_labels)

    # Get another 4 different tasks
    random_tasks = np.random.choice(random_tasks, num_random_task, replace= False)

    for random_task in random_tasks:
        corrup = random_task.split("*")[0]
        label = int(random_task.split("*")[1])
        image_file_path = os.path.join(data_path,corrup, corrup +"_"+str(label), "train_images.npy") 
        label_file_path = os.path.join(data_path,corrup, corrup +"_"+str(label), "train_labels.npy") 
        order_tasks.append(corrup +"_"+str(label))
        train_images = np.load(image_file_path)
        train_labels = np.load(label_file_path)
        train_images = train_images/255
        strain_images = train_images[:all_data_load]
        train_labels = train_labels[:all_data_load]
        strain_images= strain_images.reshape(strain_images.shape[0], -1)
        strain_images = preprocessing.normalize(strain_images)
        all_image_data.append(strain_images)
        all_label_data.append(train_labels)
     
    all_image_data = np.array(all_image_data)
    all_label_data = np.array(all_label_data)
    
    epoch_source_data = all_image_data.reshape((4, M, num_source_sample, 28*28))
    epoch_source_labels = all_label_data.reshape((4, M, num_source_sample))
    
    target_file_name = os.path.join(data_path,target_corruption, target_corruption +"_"+str(target_label), "train_images.npy")
    target_label_name = os.path.join(data_path,target_corruption, target_corruption +"_"+str(target_label), "train_labels.npy") 
    target_images = np.load(target_file_name)
    target_labels = np.load(target_label_name)
    target_images = target_images/255
    target_images= target_images.reshape(target_images.shape[0], -1)
    chosen_indices = np.random.choice(target_images.shape[0], num_target_sample, replace=False)
    target_images = target_images[chosen_indices]
    target_labels = target_labels[chosen_indices]

    return epoch_source_data, epoch_source_labels, target_images, target_labels, order_tasks, task_to_index



def generate_nu(M):
    
    nu = np.zeros(M)
    
    nu[:int(0.2 * M)] = 2
    nu[int(0.2 * M) : int(0.8 * M)] = 6
    nu[int(0.8 * M):] = 10
    
    np.random.shuffle(nu)
    nu = nu / np.sum(nu)
    
    return nu


def init_altgdmin_1(X, Y, C, M, d, k):
    """
    Spectral initialization as described in Algorithm 1 of the paper.
    
    Steps:
    1. Compute α = (C̃/Σn_m^1) * Σy_{m,n}^2
    2. Truncate: y_{m,trunc}(α) := Y_m^1 ⊙ 𝟙{|Y_m^1| ≤ √α}
    3. Compute Θ̂_0 = Σ_{m=1}^M (1/n_m^1) X_m^{1⊤} y_{m,trunc}(α) e_m⊤
    4. Get top-k singular vectors of Θ̂_0
    """
    # Calculate total number of samples across all tasks
    total_samples = sum([len(y) for y in Y])
    
    # Compute α = (C̃/Σn_m^1) * Σy_{m,n}^2 where C̃ = C (using C as C̃)
    alpha = (C / total_samples) * np.sum([np.sum(y ** 2) for y in Y])
    sqrt_alpha = np.sqrt(alpha)
    
    # Initialize Θ̂_0
    Theta_0 = np.zeros((d, M))
    
    for m in range(M):
        n_m = len(Y[m])  # Number of samples for task m
        
        # Truncate: y_{m,trunc}(α) := Y_m ⊙ 𝟙{|Y_m| ≤ √α}
        Y_trunc = Y[m].copy()
        Y_trunc[np.abs(Y[m]) > sqrt_alpha] = 0
        
        # Compute (1/n_m^1) X_m^{1⊤} y_{m,trunc}(α) e_m⊤
        # This is the m-th column of Θ̂_0
        Theta_0[:, m] = (1.0 / n_m) * X[m].dot(Y_trunc)
        
    # Get top-k singular vectors
    U_0, Sigma_0, V_0 = np.linalg.svd(Theta_0, full_matrices=False)
    U_0 = U_0[:, :k]
    Sigma_0 = Sigma_0[:k]

    return U_0, np.max(Sigma_0)


def AltGD_Min_1(X, Y, c, C, M, d, k, gd_iterations, gd_time, Theta_star=None):
    
    cumulative_time = 0
    
    start_time = time.time()
    
    U_init, Sigma_max = init_altgdmin_1(X, Y, C, M, d, k)
    
    end_time = time.time()
    
    cumulative_time = cumulative_time + end_time - start_time
    
    gd_time.append(cumulative_time)
    
    U_hat = np.copy(U_init)
    B_hat = np.zeros((k, M))
    
    error_list = []
 
    
    for num in tqdm(range(gd_iterations)):
        
        start_time = time.time()
        
        for i in range(M):
            B_hat[:, i] = np.linalg.pinv(np.dot(X[i].T, U_hat)).dot(Y[i])
            
        Theta_hat = np.dot(U_hat, B_hat)
        
        U_grad = np.zeros((d, k))
        
        for i in range(M):
            U_grad += X[i] @ (X[i].T @ Theta_hat[:, i].reshape(-1, 1) - Y[i].reshape(-1, 1)) @ B_hat[:, i].reshape(1, -1) / X[i].shape[-1]
            
        U_hat = U_hat - c * U_grad
        q, r = np.linalg.qr(U_hat)
        U_hat = q[:, :k]
        
        Theta_hat_ = np.dot(U_hat, B_hat)
        if Theta_star is not None:
            error = np.linalg.norm(Theta_star - Theta_hat_, 'fro') / np.linalg.norm(Theta_star, 'fro')
            error_list.append(error)
        
        end_time = time.time()
        
        cumulative_time = cumulative_time + end_time - start_time
        
        gd_time.append(cumulative_time)
        
    return U_hat, B_hat, gd_time, error_list


def AltGD_Min_2(X, Y, c, M, d, k, B_hat, gd_iterations):
    
    U_hat = np.copy(B_hat)
    
    B_hat = np.zeros((k, M))
    
    for num in tqdm(range(gd_iterations)):
        
        for i in range(M):
            
            B_hat[:, i] = np.linalg.pinv(np.dot(X[i].T, U_hat)).dot(Y[i])
            
        Theta_hat = np.dot(U_hat, B_hat)
            
        U_grad = np.zeros((d, k))
        
        for i in range(M):
            
            U_grad += X[i] @ (X[i].T @ Theta_hat[:, i].reshape(-1, 1) - Y[i].reshape(-1, 1)) @ B_hat[:, i].reshape(1, -1) / X[i].shape[-1]
            
        U_hat = U_hat - c * U_grad
        q, r = np.linalg.qr(U_hat)
        U_hat = q[:, :k]
        
    return U_hat, B_hat


def gd_Theta(X, Y, M, d, gd_iteration, learning_rate, Theta_star):
    
    def compute_loss_grad(X, Y, Theta, M, reg_lambda=1000):
        
        loss = 0
        grad = np.zeros_like(Theta)
        for m in range(M):
            # print("Starting GD for task:", m)
            num_sample = X[m].shape[-1]
            error_list = X[m].T.dot(Theta[:, m]) - Y[m]
            # Clip errors to prevent overflow
            error_list = np.clip(error_list, -1e6, 1e6)
            loss += 0.5 * (np.sum(error_list ** 2)) / num_sample 
            grad[:, m] = X[m].dot(error_list)
            # print("Complete GD for task:", m)
        
        # Add robust SVD for nuclear norm
        try:
            # Clip Theta values to prevent numerical issues
            Theta_clipped = np.clip(Theta, -1e6, 1e6)
            nuclear_norm = np.sum(np.linalg.svd(Theta_clipped, compute_uv=False))
            U, _, Vt = np.linalg.svd(Theta_clipped, full_matrices=False)
            nuclear_grad = np.dot(U, Vt)
        except (np.linalg.LinAlgError, RuntimeWarning):
            # If SVD fails, use simple L2 regularization instead
            nuclear_norm = 0.01 * np.sum(Theta ** 2)
            nuclear_grad = 0.01 * Theta
            
        loss += nuclear_norm
        grad += nuclear_grad
            
        return loss, grad
    
    Theta = np.ones((d, M))
    error_list = []
    error = np.linalg.norm(Theta_star - Theta, 'fro')/np.linalg.norm(Theta_star, 'fro')
    error_list.append(error)
    for i in tqdm(range(gd_iteration)):
        loss, grad = compute_loss_grad(X, Y, Theta, M)
        # Add gradient clipping to prevent explosions
        grad = np.clip(grad, -1e3, 1e3)
        Theta -= learning_rate * grad
        # Clip Theta values to prevent numerical issues
        Theta = np.clip(Theta, -1e6, 1e6)
        error = np.linalg.norm(Theta_star - Theta, 'fro')/np.linalg.norm(Theta_star, 'fro')
        error_list.append(error)
   
    return Theta, error_list


def init_altgdmin_2(X, Y, M, d, k):
    
    Z = np.zeros((d, d))
    
    for m in range(M):
        
        for n in range(X[m].shape[1]):
            
            Z += Y[m][n] ** 2 * np.dot(X[m][:, n][:, np.newaxis], X[m][:, n][:, np.newaxis].T)
            
        Z /= X[m].shape[1]
        
    Z /= m
    
    U_0, Sigma_0, V_0 = np.linalg.svd(Z, full_matrices=False)
    U_0 = U_0[:, :k]
    
    return U_0


def AltGD_Min_Collins1(X, Y, M, d, k, gd_iterations, gd_time, B_star, Theta_star):
    
    cumulative_time = 0
    
    start_time = time.time()
    
    U_init = init_altgdmin_2(X, Y, M, d, k)
    
    end_time = time.time()
    
    cumulative_time = cumulative_time + end_time - start_time
    
    gd_time.append(cumulative_time)
    
    U_hat = np.copy(U_init)
    
    B_hat = np.zeros((k, M))

    error_list = []
    error = np.linalg.norm(B_star - B_hat, 'fro')/np.linalg.norm(B_star, 'fro')
    error_list.append(error)
    
    for num in tqdm(range(gd_iterations)):
        
        start_time = time.time()
        
        M_sub_value = int(np.random.uniform(low = 1e-1, high = 1 + 1e-10) * M)
        M_sub = np.random.choice(range(M), M_sub_value, replace = False)
        
        for i in M_sub:
            
            XB = X[i].T.dot(U_hat)
            B_hat[:, i] = np.linalg.pinv(XB.T.dot(XB)).dot(XB.T).dot(Y[i])
            
        Theta_hat = np.dot(U_hat, B_hat)
            
        U_grad = np.zeros((d, k))
        
        for i in M_sub:
            
            U_grad += X[i] @ (X[i].T @ Theta_hat[:, i].reshape(-1, 1) - Y[i].reshape(-1, 1)) @ B_hat[:, i].reshape(1, -1) / X[i].shape[-1]
            
        U_hat = U_hat - (1 / M_sub_value) * U_grad
        q, r = np.linalg.qr(U_hat)
        U_hat = q[:, :k]
        
        Theta_hat_ = np.dot(U_hat, B_hat)
        error = np.linalg.norm(Theta_star - Theta_hat_, 'fro') / np.linalg.norm(Theta_star, 'fro')
        error_list.append(error)
        end_time = time.time()
        
        cumulative_time = cumulative_time + end_time - start_time
        
        gd_time.append(cumulative_time)
        
    return U_hat, B_hat, gd_time,error_list


def AltGD_Min_Collins2(X, Y, M, d, k, B_hat, gd_iterations):
    
    U_hat = np.copy(B_hat)
    
    B_hat = np.zeros((k, M))
    
    for num in tqdm(range(gd_iterations)):
        M_sub_value = int(np.random.uniform(low = 1e-1, high = 1 + 1e-10) * M)
        M_sub = np.random.choice(range(M), M_sub_value, replace = False)
        
        for i in M_sub:
            
            XB = X[i].T.dot(U_hat)
            B_hat[:, i] = np.linalg.pinv(XB.T.dot(XB)).dot(XB.T).dot(Y[i])
            
        Theta_hat = np.dot(U_hat, B_hat)
            
        U_grad = np.zeros((d, k))
        
        for i in M_sub:
            
            U_grad += X[i] @ (X[i].T @ Theta_hat[:, i].reshape(-1, 1) - Y[i].reshape(-1, 1)) @ B_hat[:, i].reshape(1, -1) / X[i].shape[-1]
        
        U_hat = U_hat - (1 / M_sub_value) * U_grad
        q, r = np.linalg.qr(U_hat)
        U_hat = q[:, :k]
        
    return U_hat, B_hat


def plot_ER(data, d_list, trials, epochs):
    
    x_value = np.array(d_list)
    y_value = np.zeros((len(d_list), epochs))
    
    for i, d in enumerate(d_list):
        
        for T in range(trials * len(d_list)):
            
            if data[T][1] == d:
                
                y_value[i] += data[T][-1]
                
    y_value /= trials
    
    return x_value, y_value


def plot_error_gd(data, gd_iterations, trials):
    
    x_value = np.array([i for i in range(gd_iterations + 1)])
    y_value = np.zeros(gd_iterations + 1)

    for i in range(gd_iterations + 1):

        for T in range(trials * len(d_list)):

            if data[T][1] == 3:

                y_value[i] += data[T][-1][i]

    y_value = y_value / trials
            
    return x_value, y_value


def plot_error_epoch(data, epochs, trials):
    
    x_value = np.array([i for i in range(epochs)])
    y_value = np.zeros((len(d_list), epochs))

    for index, d in enumerate(d_list):

        for i in range(epochs):

            for T in range(trials * len(d_list)):

                if data[T][1] == d:

                    y_value[index, i] += data[T][-1][i]
                    
    y_value = y_value / trials
            
    return x_value, y_value


def plot_error_epoch_MOM(data, epochs, trials):
    
    x_value = np.array([i for i in range(epochs)])
    y_value = np.zeros((len(d_list), epochs))

    for index, d in enumerate(d_list):

        for i in range(epochs):

            for T in range(trials * len(d_list)):

                if data[T][1] == d:

                    y_value[index, i] += data[T][-1]
                    
    y_value = y_value / trials
            
    return x_value, y_value

