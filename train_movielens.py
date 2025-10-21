import os
import time
import json
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.linalg import orth
from scipy.optimize import minimize
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans
from utils import *
from nn_theta_star import ThetaStarNN

np.random.seed(42)


class CollaborativeFiltering():
    def __init__(self, rating_matrix, n, item_based=True):
        """
        rating_matrix: [n_users x n_items] matrix
        n: predict ratings based on `n` most similar items/users
        item_based: if True, then item-item cf is used otherwise user-user cf is used (default: True)
        """
        self.rating_matrix = rating_matrix
        self.n_users = self.rating_matrix.shape[0]
        self.n_items = self.rating_matrix.shape[1]
        self.n = n
        self.item_based = item_based

    def pearson_correlation_sim(self, id1, id2):
        """
        calculative pearson correlation similarity score for item/user id1, id2 (index starts with 0)
        """
        if self.item_based:
            ratings_x = self.rating_matrix[:,id1]
            ratings_y = self.rating_matrix[:,id2]
        else:
            ratings_x = self.rating_matrix[id1,:]
            ratings_y = self.rating_matrix[id2,:]

        common_ratings = np.nonzero(np.multiply(ratings_x, ratings_y))

        if len(common_ratings) == 0:
            return 0

        rx_minus_mean = ratings_x[common_ratings] - ratings_x.mean()
        ry_minus_mean = ratings_y[common_ratings] - ratings_y.mean()

        denominator_val = (np.linalg.norm(rx_minus_mean) * np.linalg.norm(ry_minus_mean))
        if denominator_val == 0:
            return 0

        return np.sum(np.multiply(rx_minus_mean, ry_minus_mean)) / denominator_val

    def n_most_similar_items(self, item_id, rated_by_user_id):
        """
        find `n` most similar items to item `item_id` (index starts from 0) based on pearson correlation similarity which are rated by user `rated_by_user_id`
        return: ids of `n` most similar items to item `item_id`, their respective similarity scores
        """
        # similarities is list of [id,sim] for those items rated by `user_id`. sim is similarity of `item_id` and id
        similarities = [ [id2,self.pearson_correlation_sim(item_id,id2)] for id2 in range(self.n_items) if not self.rating_matrix[rated_by_user_id, id2] == 0 and not id2 == item_id ]
        similarities = np.array(sorted(similarities, key = lambda x: x[1], reverse=True))[:self.n]

        if similarities.shape[0] == 0: # no similar found
            return [],[]

        n_most_similar = similarities[:,0].astype(int)
        similarity_scores = similarities[:,1]

        return n_most_similar, similarity_scores

    def n_most_similar_users(self, user_id, rated_item_id):
        """
        find `n` most similar users to item `user_id` (index starts from 0) based on pearson correlation similarity who have rated item `rated_item_id`
        return: ids of `n` most similar users to user `user_id`, their respective similarity scores
        """
        similarities = [ [id2,self.pearson_correlation_sim(user_id,id2)] for id2 in range(self.n_users) if not self.rating_matrix[id2, rated_item_id] == 0 and not id2 == user_id ]
        similarities = np.array(sorted(similarities, key = lambda x: x[1], reverse=True))[:self.n]
        
        if similarities.shape[0] == 0: # no similar found
            return [],[]

        n_most_similar = similarities[:,0].astype(int)
        similarity_scores = similarities[:,1]

        return n_most_similar, similarity_scores

    def predict_rating(self, user_id, item_id):
        """
        predict rating of `user_id` to `item_id` based on weighted avg of `n` most similar items/users
        """
        if self.item_based:
            n_most_similar, similarity_scores = self.n_most_similar_items(item_id, user_id)
            if sum(similarity_scores) == 0:
                return 0
            ratings_of_n_most_similar = self.rating_matrix[user_id,n_most_similar] # ratings by `user_id` for n most similar items to `item_id`
        else:
            n_most_similar, similarity_scores = self.n_most_similar_users(user_id, item_id)
            if sum(similarity_scores) == 0:
                return 0
            ratings_of_n_most_similar = self.rating_matrix[n_most_similar, item_id] # ratings by n most similar users for `item_id`

        predicted_rating = np.average(ratings_of_n_most_similar, weights=similarity_scores)
        return predicted_rating

    def calc_rmse(self, test_ratings):
        """
        test_ratings: [n_test_ratings x 3] array where each row of the form [user_id, item_id, r]
        """
        actual = test_ratings[:,2]
        predicted = np.array([self.predict_rating(uid,iid) for uid,iid,_ in test_ratings])

        rmse = np.sqrt(np.mean((actual - predicted)**2))
        return rmse

def load_rating_matrix(data_dir="data/ml-100k"):
    """Load MovieLens rating matrix from ua.base file."""
    ua_base_file = os.path.join(data_dir, "ua.base")
    df = pd.read_csv(ua_base_file, sep='\t', header=None, 
                     names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    n_users = df['user_id'].max()
    n_movies = df['movie_id'].max()
    
    rating_matrix = np.zeros((n_users, n_movies))
    
    for _, row in df.iterrows():
        user_id = int(row['user_id']) - 1 
        movie_id = int(row['movie_id']) - 1 
        rating = row['rating']
        rating_matrix[user_id, movie_id] = rating
    
    return rating_matrix

def collaborative_filtering_fill_custom(matrix, num_factors=20, num_iterations=50, lr=0.01, reg=0.1):
    """Custom matrix factorization for collaborative filtering."""
    n_users, n_movies = matrix.shape
    
    U = np.random.normal(0, 0.1, (n_users, num_factors))
    V = np.random.normal(0, 0.1, (n_movies, num_factors))
    
    observed = matrix > 0
    
    for iteration in range(num_iterations):
        for u in range(n_users):
            for m in range(n_movies):
                if not observed[u, m]:
                    continue
                r = matrix[u, m]
                pred = np.dot(U[u], V[m])
                err = r - pred
                U[u] += lr * (err * V[m] - reg * U[u])
                V[m] += lr * (err * U[u] - reg * V[m])
        
        if (iteration + 1) % 10 == 0:
            predicted = np.dot(U, V.T)
            rmse = np.sqrt(np.mean((matrix[observed] - predicted[observed])**2))
    
    completed_matrix = np.dot(U, V.T)
    completed_matrix[observed] = matrix[observed]
    
    return completed_matrix


def generate_action_set(M, d):
    complete_rating = np.load("data/estimated_matrix.npy")
    complete_rating = complete_rating / 5 
 
    model = NMF(n_components = int(np.sqrt(d)), init = 'nndsvda', max_iter = 10000)
    W = model.fit_transform(complete_rating)
    H = model.components_
 
    theta_true = np.eye(int(np.sqrt(d))).ravel()
    Theta = np.tile(theta_true, (M, 1)).T
    Theta[:, M - 2] = np.ones(d) / np.sqrt(d)
    Theta[:, M - 1] = np.ones(d) / np.sqrt(d)
 
    kmeans = KMeans(n_clusters = M, random_state = 0).fit(H.T)
    labels = kmeans.labels_
 
    Action_list = np.empty((W.shape[0], M), dtype = object)
 
    for t in range(W.shape[0]):
 
        for i in range(M):

            columns_cluster = np.where(labels == i)[0]
            size = len(columns_cluster)
            Action_list[t, i] = np.empty((size, d), dtype = float)
 
            for ac, column_idx in enumerate(columns_cluster):
                
                outer = np.outer(W[t], H[:, column_idx])

                if i == M - 2 or i == M - 1:

                    diag_values = W[t] * H[:, column_idx]
                    Action_list[t, i][ac] = np.tile(diag_values[:, None], (1, int(np.sqrt(d)))).ravel()

                else:

                    Action_list[t, i][ac] = outer.ravel()
                    
    return Action_list, Theta


def get_config_filename(algo, params):
    """
    Generate config filename similar to MNIST-C.
    """
    target_corruption = params["target_corruption"]
    target_label = params["target_label"]
    M = params["M"]
    k = params["k"]
    
    if algo == "altGD_ada":
        return f"AltGD_ADA_{target_corruption}_{target_label}_numtasks_{M}_k_{k}.json"
    elif algo == "altGD_no_ada":
        return f"AltGD_NOADA_{target_corruption}_{target_label}_numtasks_{M}_k_{k}.json"
    elif algo == "chen":
        return f"Chen_ADA_{target_corruption}_{target_label}_numtasks_{M}_k_{k}.json"
    elif algo == "mom":
        return f"MOM_{target_corruption}_{target_label}_numtasks_{M}_k_{k}.json"
    else:
        return f"{algo}_{target_corruption}_{target_label}_numtasks_{M}_k_{k}.json"


def run_movielens_bandit_experiment():
    algorithms = ["altGD_ada", "altGD_no_ada", "chen", "mom", "collins"]  
    k_list = [2] 
    M = 80  
    d = 100  
    target_cluster = M - 1  
    
    Action_list, Theta_star = generate_action_set(M, d)
    
    all_results = {}
    
    for k in k_list:
        params = {}
        params["d"] = d
        params["k"] = k
        params["M"] = M
        params["epochs"] = 4
        params["C"] = 3
        params["c"] = 0.05
        params["gd_iterations"] = 1000  
        params["increase_gd_iteration"] = 10
        params["num_source_sample"] = 2370 
        params["num_target_sample"] = 20
        params["samples_per_task"] = params["num_source_sample"] // (M - 1) 
        params["learning_rate"] = 1e-6  
        params["target_corruption"] = "movielens_cluster"
        params["target_label"] = target_cluster
        
        epoch_source_data, epoch_source_labels, target_data, target_labels, order_tasks = generate_movielens_task_data(
            Action_list, Theta_star, params
        )
        
        num_source_tasks = epoch_source_data.shape[1]  
        params["M"] = num_source_tasks 
        
        source_tasks = [i for i in range(M) if i != target_cluster]
        Theta_star_source = Theta_star[:, source_tasks] 

        k_results = {}
        
        for algo in algorithms:
            print(f"\n{'-'*30}")
            print(f"Running {algo.upper()}")
            print(f"{'-'*30}")
            
            if algo == "altGD_ada":
                run_alt_gd_min_ada(
                    epoch_source_data=epoch_source_data,
                    epoch_source_labels=epoch_source_labels,
                    target_data=target_data,
                    target_labels=target_labels,
                    order_tasks=order_tasks,
                    params=params.copy(),
                    Theta_star=Theta_star_source  
                )
                
            elif algo == "altGD_no_ada":
                run_alt_gd_min_noada(
                    epoch_source_data=epoch_source_data,
                    epoch_source_labels=epoch_source_labels,
                    target_data=target_data,
                    target_labels=target_labels,
                    order_tasks=order_tasks,
                    params=params.copy(),
                    Theta_star=Theta_star_source  
                )
                
            elif algo == "chen":
                run_chen(
                    epoch_source_data=epoch_source_data,
                    epoch_source_labels=epoch_source_labels,
                    target_data=target_data,
                    target_labels=target_labels,
                    order_tasks=order_tasks,
                    params=params.copy(),
                    Theta_star=Theta_star_source  
                )
                
            elif algo == "mom":
                run_mom(
                    epoch_source_data=epoch_source_data,
                    epoch_source_labels=epoch_source_labels,
                    target_data=target_data,
                    target_labels=target_labels,
                    order_tasks=order_tasks,
                    params=params.copy(),
                    Theta_star=Theta_star_source  
                )
                
            elif algo == "collins":
                run_Collins(
                    epoch_source_data=epoch_source_data,
                    epoch_source_labels=epoch_source_labels,
                    target_data=target_data,
                    target_labels=target_labels,
                    order_tasks=order_tasks,
                    params=params.copy()
                )
                
            config_file = f"config/{get_config_filename(algo, params)}"
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    result_data = json.load(f)
                    if 'ER' in result_data:
                        k_results[algo] = result_data['ER']
                        print(f"{algo} completed. ER saved to config.")
                    else:
                        print(f"Warning: No ER found in {config_file}")
            else:
                print(f"Warning: Config file {config_file} not found")
        
        all_results[f"k_{k}"] = k_results
    
    print("\n" + "="*60)
    print("MOVIELENS BANDIT EXPERIMENT RESULTS")
    print("="*60)
    
    for k_key, k_results in all_results.items():
        print(f"\n{k_key.upper()}:")
        for algo, er_result in k_results.items():
            if er_result is not None:
                if isinstance(er_result, list) and len(er_result) > 0:
                    if isinstance(er_result[0], tuple):
                        final_er = er_result[0][1][-1] if er_result[0][1] else "N/A"
                    else:
                        final_er = er_result[-1] if er_result else "N/A"
                    print(f"  {algo}: Final ER = {final_er}")
                else:
                    print(f"  {algo}: ER = {er_result}")
            else:
                print(f"  {algo}: Failed")
    
    print(f"\nConfig files saved in config/ directory")
    print("="*60)
    
    return all_results


def generate_movielens_task_data(Action_list, Theta_star, params):
    epochs = params["epochs"]
    M = params["M"]
    num_source_sample = params["num_source_sample"]
    num_target_sample = params["num_target_sample"]
    d = params["d"]
    target_cluster = params["target_label"]
    
    np.random.seed(23)
    
    n_users = Action_list.shape[0] 
    
    source_tasks = [i for i in range(M) if i != target_cluster]
    num_source_tasks = len(source_tasks)  # Should be M-1 = 79
    samples_per_task = params["samples_per_task"]  # Samples per task
    
    order_tasks = [f"cluster_{i}" for i in source_tasks]
    
    epoch_source_data = np.zeros((epochs, num_source_tasks, samples_per_task, d))
    epoch_source_labels = np.zeros((epochs, num_source_tasks, samples_per_task))
    
    task_samples = {}
    
    for task_idx, cluster_id in enumerate(source_tasks):
        cluster_samples = []
        cluster_labels = []
        cluster_theta = Theta_star[:, cluster_id]  
        
        for _ in range(samples_per_task):
            user_idx = np.random.choice(n_users)
            
            user_cluster_actions = Action_list[user_idx, cluster_id]
            
            if hasattr(user_cluster_actions, '__len__') and len(user_cluster_actions) > 0:
                action_idx = np.random.choice(len(user_cluster_actions))
                selected_action = user_cluster_actions[action_idx]
            else:
                selected_action = np.random.randn(d) * 0.1
            
            if len(selected_action) != d:
                selected_action = np.pad(selected_action, (0, max(0, d - len(selected_action))))[:d]
            
            # Compute label: Y_i = X_i * theta_star_i
            y_val = np.dot(selected_action, cluster_theta)
            
            cluster_samples.append(selected_action)
            cluster_labels.append(y_val)
        
        task_samples[task_idx] = {
            'samples': np.array(cluster_samples),  
            'labels': np.array(cluster_labels)     
        }
    
    # Fill epoch data arrays with identical data for all epochs (like MNIST-C)
    for epoch in range(epochs):
        for task_idx in range(num_source_tasks):
            epoch_source_data[epoch, task_idx] = task_samples[task_idx]['samples']
            epoch_source_labels[epoch, task_idx] = task_samples[task_idx]['labels']
    
    np.random.seed(42)  # Reset seed for target data consistency
    target_data = np.zeros((num_target_sample, d))
    target_labels = np.zeros(num_target_sample)
    target_theta = Theta_star[:, target_cluster]
    
    for sample_idx in range(num_target_sample):
        user_idx = np.random.choice(n_users)
        
        user_cluster_actions = Action_list[user_idx, target_cluster]
        
        if hasattr(user_cluster_actions, '__len__') and len(user_cluster_actions) > 0:
            action_idx = np.random.choice(len(user_cluster_actions))
            selected_action = user_cluster_actions[action_idx]
        else:
            selected_action = np.random.randn(d) * 0.1
        
        # Ensure correct shape
        if len(selected_action) != d:
            selected_action = np.pad(selected_action, (0, max(0, d - len(selected_action))))[:d]
        
        target_data[sample_idx] = selected_action
        target_labels[sample_idx] = np.dot(selected_action, target_theta)
    
    return epoch_source_data, epoch_source_labels, target_data, target_labels, order_tasks


def run_all_target_experiments():
    """
    Run MovieLens experiment with last task (task 29) as target and remaining 29 tasks as source.
    """
    print("="*80)
    print("RUNNING MOVIELENS EXPERIMENT - TASK 29 AS TARGET")
    print("="*80)
    
    try:
        results = run_movielens_bandit_experiment()
        print(f"✅ Experiment completed successfully with task 29 as target")
        return {"target_29": results}
        
    except Exception as e:
        print(f"❌ Error in experiment: {str(e)}")
        return {"target_29": {"error": str(e)}}


# Run the MovieLens experiment
if __name__ == "__main__":
    # Run experiments for all 4 target clusters
    all_results = run_all_target_experiments()
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
