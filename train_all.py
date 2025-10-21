import os
import time
import json
import torch 
import torch.optim as optim
import numpy as np
from scipy.linalg import orth
from utils import *
from scipy.optimize import minimize
from nn_theta_star import ThetaStarNN
np.random.seed(42)


target_tasks = [ "brightness*0"]
algorithms = ["altGD_ada", "chen", "mom" ,"altGD_no_ada", "Collins"]
num_random_tasks = [16]
k_list = [30]

for target_task in target_tasks:
    params = {}
    target_corruption = target_task.split("*")[0]
    target_label = int(target_task.split("*")[1])
    num_target_sample = 50
    num_source_sample = 1500
    all_data_load = 6000
    params["target_corruption"] = target_corruption
    params["target_label"] = target_label
    params["gd_iterations"] = 1000
    params["increase_gd_iteration"]= 10
    params["num_source_sample"] = num_source_sample
    params["num_target_sample"] = num_target_sample

    params["d"] = 784
    params["epochs"] = 4
    params["C"] = 1000
    params["all_data_load"] = 6000
    params["samples_per_epoch"] = 100
    
    X_source_temp, Y_source_temp, X_target_temp, Y_target_temp, _, _ = data_loader_related_tasks(target_corruption, target_label, 135, 1500, 1500, all_data_load, 159)
    X_all = X_source_temp.reshape(159, all_data_load, 784)
    Y_all = Y_source_temp.reshape(159, all_data_load)
    X_source = [torch.tensor(X_all[m], dtype=torch.float32) for m in range(159)]
    Y_source = [torch.tensor(Y_all[m], dtype=torch.float32) for m in range(159)]
    X_target = torch.tensor(X_target_temp, dtype=torch.float32)
    Y_target = torch.tensor(Y_target_temp, dtype=torch.float32)
    
    for k in k_list:
            
        params["k"] = k
        model = ThetaStarNN(params["d"], k, 40)
        optimizer = optim.SGD(model.parameters(), lr = 0.1)
        
        for epoch in range(10):
            
            optimizer.zero_grad()
            loss = model.source_loss(X_source, Y_source)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            
        B_star = model.linear.weight
        B_star = B_star.detach().numpy().T
        W_star = np.array([np.array(w.data) for w in model.W])
        W_star = W_star.T
        w_target_star = np.array(model.w_target.data)
        
        for num_task in num_random_tasks:
            
            M  = 24 + num_task
            params["M"] = M
            params["c"] = 0.1 / M
            params["learning_rate"] = 0.1 / M
            epoch_source_data, epoch_source_labels, target_data, target_labels, order_tasks,_ = data_loader_related_tasks(target_corruption, target_label, num_task, num_target_sample, num_source_sample, all_data_load, M)
            noise = np.random.normal(0, 1e-5, size = (params["epochs"], params["M"] + 1, params["num_source_sample"]))
            
            for algo in algorithms:
                
                print("Training ", algo, "for target task", target_task)
                print("Number of random tasks:", num_task)
                print("Rank k:", k)
                
                if algo == "altGD_ada":
                    
                    run_alt_gd_min_ada( epoch_source_data = epoch_source_data, 
                                        epoch_source_labels=epoch_source_labels, 
                                        target_data = target_data, 
                                        target_labels=target_labels, 
                                        order_tasks=order_tasks,
                                        params= params, noise = noise,
                                        B_star = B_star,
                                        W_star = W_star,
                                        w_target_star = w_target_star
                                        )
                
                if algo == "altGD_no_ada":
                    
                    run_alt_gd_min_noada(epoch_source_data = epoch_source_data, 
                                    epoch_source_labels=epoch_source_labels, 
                                    target_data = target_data, 
                                    target_labels=target_labels, 
                                    order_tasks=order_tasks,
                                    params= params, noise = noise,
                                    B_star = B_star,
                                    W_star = W_star,
                                    w_target_star = w_target_star)
                    
                if algo == "chen":
                    
                    run_chen(epoch_source_data = epoch_source_data, 
                            epoch_source_labels=epoch_source_labels, 
                            target_data = target_data, 
                            target_labels=target_labels, 
                            order_tasks=order_tasks,
                            params= params, noise = noise,
                            B_star = B_star,
                            W_star = W_star,
                            w_target_star = w_target_star)
                    
                if algo == "mom":
                    
                    run_mom(epoch_source_data = epoch_source_data,
                            epoch_source_labels=epoch_source_labels,
                            target_data = target_data,
                            target_labels=target_labels,
                            order_tasks=order_tasks,
                            params= params, noise = noise,
                            B_star = B_star,
                            W_star = W_star,
                            w_target_star = w_target_star)
                    
                if algo == "Collins":
                    
                    run_Collins( epoch_source_data = epoch_source_data, 
                                        epoch_source_labels=epoch_source_labels, 
                                        target_data = target_data, 
                                        target_labels=target_labels, 
                                        order_tasks=order_tasks,
                                        params= params, noise = noise,
                                        B_star = B_star,
                                        W_star = W_star,
                                        w_target_star = w_target_star)
                
