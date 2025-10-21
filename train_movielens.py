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
    """
    Generate action set using NMF decomposition and clustering approach.
    
    Args:
        M: Total number of tasks (including target)
        d: Feature dimensionality
    
    Returns:
        user_features: User feature matrix W (n_users x d)
        Theta_star: Task parameter matrix (d x M) where each column is a task's theta
        movie_cluster_assignments: Which cluster each movie belongs to
        target_theta: Target task theta (linear combination of 2 source tasks)
    """
    complete_rating = np.load("data/estimated_matrix.npy")
    complete_rating = complete_rating / 5.0  
    n_users, n_items = complete_rating.shape
    
    nmf = NMF(n_components=d, init='random', random_state=42, max_iter=200)
    W = nmf.fit_transform(complete_rating)  
    H = nmf.components_.T  
    
    
    
    num_source_tasks = M - 1  
    kmeans = KMeans(n_clusters=num_source_tasks, random_state=42, n_init=10)
    movie_cluster_assignments = kmeans.fit_predict(H)

    cluster_centroids = kmeans.cluster_centers_

    # Find any two linearly independent source tasks
    def find_independent_tasks(centroids):
        """
        Find any two tasks whose theta vectors are linearly independent.
        Returns the indices of the first pair found that are linearly independent.
        """
        num_tasks = centroids.shape[0]
        
        # Check pairs starting from the beginning
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                theta_i = centroids[i]  # Shape: (d,)
                theta_j = centroids[j]  # Shape: (d,)
                
                # Create matrix with the two vectors as columns
                matrix = np.column_stack([theta_i, theta_j])  # Shape: (d, 2)
                
                # Check if they are linearly independent
                rank = np.linalg.matrix_rank(matrix)
                
                if rank == 2:  # Both vectors are linearly independent
                    return i, j
        
        # Fallback: if no independent pair found, return first two
        print("Warning: No linearly independent pair found, using tasks 0 and 1")
        return 0, 1
    
    # Find any linearly independent pair
    source_task_1, source_task_2 = find_independent_tasks(cluster_centroids)
    
    print(f"Selected linearly independent tasks: {source_task_1} and {source_task_2}")
    
    # Verify linear independence
    theta_1 = cluster_centroids[source_task_1]
    theta_2 = cluster_centroids[source_task_2]
    matrix = np.column_stack([theta_1, theta_2])
    rank = np.linalg.matrix_rank(matrix)

    
    alpha = 0.9
    beta = 0.1
    target_theta = alpha * cluster_centroids[source_task_1] + beta * cluster_centroids[source_task_2]
    

    Theta_star = np.zeros((d, M))
    Theta_star[:, :num_source_tasks] = cluster_centroids.T  
    Theta_star[:, -1] = target_theta  

    # print(Theta_star)
    
    
    return W, Theta_star, movie_cluster_assignments, {
        'source_task_1': source_task_1,
        'source_task_2': source_task_2,
        'alpha': alpha,
        'beta': beta,
        'target_theta': target_theta
    }


def get_config_filename(algo, params):
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
    
    M = 30 
    d = 10 
    
    # Generate action set using NMF and clustering
    user_features, Theta_star, movie_cluster_assignments, target_info = generate_action_set(M, d)
    
    # Rank k is determined by the rank of Theta_star
    # k = np.linalg.matrix_rank(Theta_star)
    k = 2
    print(f"Using rank k = {k} (determined by matrix rank of Theta_star)")
    
    # Setup parameters
    params = {}
    params["d"] = int(d)  
    params["k"] = int(k)  
    params["M"] = int(M - 1)  
    params["epochs"] = 4
    params["C"] = 3
    params["c"] = 0.1
    params["gd_iterations"] = 500  
    params["increase_gd_iteration"] = 10
    params["num_source_sample"] = 1450  # Total source samples across all tasks
    params["num_target_sample"] = 20
    params["samples_per_task"] = int(params["num_source_sample"] // (M - 1))  # Samples per source task
    params["learning_rate"] = 1e-4  # Increased learning rate for better convergence
    params["target_corruption"] = "movielens_nmf"
    params["target_label"] = "target"
    
    # Pass target theta info to params for consistent evaluation
    params["target_theta"] = target_info['target_theta'].tolist()
    params["target_info"] = {
        'source_task_1': int(target_info['source_task_1']),
        'source_task_2': int(target_info['source_task_2']), 
        'alpha': float(target_info['alpha']),
        'beta': float(target_info['beta'])
    }
    
    print(f"Generating task data for k={k}...")
    epoch_source_data, epoch_source_labels, target_data, target_labels, order_tasks = generate_movielens_task_data(
        user_features, Theta_star, movie_cluster_assignments, params
    )
    
    # Use only source tasks for Theta_star (exclude target)
    Theta_star_source = Theta_star[:, :-1]  # All columns except last (target)
    
    results = {}
    
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
                    results[algo] = result_data['ER']
                    print(f"{algo} completed. ER saved to config.")
                else:
                    print(f"Warning: No ER found in {config_file}")
        else:
            print(f"Warning: Config file {config_file} not found")
    
    
    print("\n" + "="*60)
    print("MOVIELENS NMF-BASED EXPERIMENT RESULTS")
    print("="*60)
    
    # Print target task information
    print(f"\nTarget Task Info:")
    print(f"  Created as linear combination of source tasks {target_info['source_task_1']} and {target_info['source_task_2']}")
    print(f"  Combination: {target_info['alpha']} * θ_{target_info['source_task_1']} + {target_info['beta']} * θ_{target_info['source_task_2']}")
    print(f"  Using rank k = {k}")
    
    print(f"\nRESULTS:")
    for algo, er_result in results.items():
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
    
    return results
def generate_movielens_task_data(user_features, Theta_star, movie_cluster_assignments, params):
    """
    Generate task data using NMF-based approach.
    
    Args:
        user_features: User feature matrix W (n_users x d)
        Theta_star: Task parameter matrix (d x M) 
        movie_cluster_assignments: Which cluster each movie belongs to
        params: Experiment parameters
    
    Returns:
        epoch_source_data, epoch_source_labels, target_data, target_labels, order_tasks
    """
    epochs = params["epochs"]
    M = params["M"]  # Number of source tasks
    num_source_sample = params["num_source_sample"]
    num_target_sample = params["num_target_sample"]
    d = params["d"]
    samples_per_task = params["samples_per_task"]
    
    np.random.seed(42)
    
    n_users, d_features = user_features.shape
    print(f"Generating task data from {n_users} users with {d_features} features")
    
    # Create order_tasks list for source tasks
    order_tasks = [f"nmf_cluster_{i}" for i in range(M)]
    
    # Initialize epoch data arrays
    epoch_source_data = np.zeros((epochs, M, samples_per_task, d))
    epoch_source_labels = np.zeros((epochs, M, samples_per_task))
    
    print(f"Generating data for {M} source tasks with {samples_per_task} samples each...")
    
    # Generate source task data
    for task_idx in range(M):
        task_theta = Theta_star[:, task_idx]  # Get theta for this source task
        
        # Generate samples for this task
        task_samples = []
        task_labels = []
        
        for sample_idx in range(samples_per_task):
            # Randomly select a user
            user_idx = np.random.choice(n_users)
            user_feature = user_features[user_idx]  # Shape: (d,)
            
            # Generate label: y = x^T * theta + noise
            y_val = np.dot(user_feature, task_theta)
            
            # Add small amount of noise
            noise = np.random.normal(0, 0.01)
            y_val += noise
            
            task_samples.append(user_feature)
            task_labels.append(y_val)
        
        # Convert to arrays
        task_samples = np.array(task_samples)  # Shape: (samples_per_task, d)
        task_labels = np.array(task_labels)    # Shape: (samples_per_task,)
        
        # Fill all epochs with the same data (consistent with MNIST-C approach)
        for epoch in range(epochs):
            epoch_source_data[epoch, task_idx] = task_samples
            epoch_source_labels[epoch, task_idx] = task_labels
    
    print(f"Generating target task data with {num_target_sample} samples...")
    
    # Generate target task data (using the last column of Theta_star as target theta)
    target_theta = Theta_star[:, -1]  # Last column is target task theta
    target_data = np.zeros((num_target_sample, d))
    target_labels = np.zeros(num_target_sample)
    
    for sample_idx in range(num_target_sample):
        # Randomly select a user
        user_idx = np.random.choice(n_users)
        user_feature = user_features[user_idx]  # Shape: (d,)
        
        # Generate label: y = x^T * theta + noise
        y_val = np.dot(user_feature, target_theta)
        
        # Add small amount of noise
        noise = np.random.normal(0, 0.01)
        y_val += noise
        
        target_data[sample_idx] = user_feature
        target_labels[sample_idx] = y_val
    
    print(f"Task data generation completed:")
    print(f"  Source data shape: {epoch_source_data.shape}")
    print(f"  Source labels shape: {epoch_source_labels.shape}")
    print(f"  Target data shape: {target_data.shape}")
    print(f"  Target labels shape: {target_labels.shape}")
    
    return epoch_source_data, epoch_source_labels, target_data, target_labels, order_tasks


def run_all_target_experiments():
    """
    Run MovieLens NMF-based experiment with adaptive sampling.
    """
    print("="*80)
    print("RUNNING MOVIELENS NMF-BASED EXPERIMENT")
    print("Uses NMF decomposition + K-means clustering + linear combination target")
    print("="*80)
    
    try:
        results = run_movielens_bandit_experiment()
        print(f"✅ NMF-based experiment completed successfully")
        return {"nmf_experiment": results}
        
    except Exception as e:
        print(f"❌ Error in NMF-based experiment: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"nmf_experiment": {"error": str(e)}}


# Run the MovieLens NMF-based experiment
if __name__ == "__main__":
    all_results = run_all_target_experiments()
