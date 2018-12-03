###############################################################################

import time
import numpy as np
import random as rn
from itertools import product as cart

from config_helper import ConfigHelper



class KCM_FGH:
	# Class which applies the KCM-F-GH algorithm

	sigma2 = None			# "Variance" of the gaussian
	n = None				# Number of objects
	p = None				# Number of variables
	s = None				# Inverse of the global width hyperparams
	partition = None		# Objects and their clusters
	clusters = None			# Clusters and their objects
	score = None			# Objective function score
	X = None				# Rows of data and features
	nb_iterations = None	# Number of iterations
	converged = None		# Whether the algorithm converged or not

	c = 7					# Number of clusters

	@staticmethod
	def _create_run_result(view_type, e):
		# Stores the results of a single run for the KCM-F-GH

		result = {}
		result["view_type"] = view_type
		result["execution"] = e
		result["iterations"] = KCM_FGH.nb_iterations
		result["hyperparams"] = KCM_FGH.s
		result["partition"] = KCM_FGH.partition
		result["objective_score"] = KCM_FGH.score
		result["converged"] = KCM_FGH.converged
		return result

	@staticmethod
	def _initialize_clusters():
		# Initialize the clusters with single randomly selected prototypes

		sample = rn.sample(range(KCM_FGH.n), KCM_FGH.c)
		KCM_FGH.clusters = np.array([np.array([i]) for i \
			in sample])

	@staticmethod
	def _allocate_in_clusters():
		# Allocate instances in the calculated clusters

		KCM_FGH.clusters = [[i for i in range(KCM_FGH.n) \
				if cluster_idx==KCM_FGH.partition[i]] \
				for cluster_idx in range(KCM_FGH.c)]

	@staticmethod
	def _euclidean(i, j):
		# Calculate the euclidean distance between two examples

		distance_vector = KCM_FGH.X[i] - KCM_FGH.X[j]
		squared_distance_vector = distance_vector**2
		distance = np.sum(squared_distance_vector)
		return np.sqrt(distance)

	@staticmethod
	def _initialize_sigma2():
		# Initialize the sigma² value

		all_idxs = range(KCM_FGH.n)
		cartesian = cart(all_idxs, all_idxs)
		distances = [KCM_FGH._euclidean(i, j) for (i, j) in cartesian \
						 if i < j]
		quantiles = np.percentile(distances, [10, 90])
		KCM_FGH.sigma2 = np.mean(quantiles)

	@staticmethod
	def _dist_to_cluster(cluster):
		# Calculate distance of examples to a single cluster

		Pi = len(cluster)

		kernel_values_term_1 = [[KCM_FGH._kernel(KCM_FGH.X[k], KCM_FGH.X[l]) \
						 	for l in cluster] for k in range(KCM_FGH.n)]
		first_term_sums = np.array([np.sum(values) for values \
								in kernel_values_term_1])

		first_term = first_term_sums/Pi

		cluster_pairs = cart(cluster, repeat=2)
		kernel_values_term_2 = [KCM_FGH._kernel(KCM_FGH.X[r], KCM_FGH.X[s]) \
							for (r, s) in cluster_pairs if r < s]
		second_term_sum = np.sum(kernel_values_term_2)
		second_term = second_term_sum/(Pi**2.0)

		return 1.0 - 2.0*first_term + second_term

	@staticmethod
	def _representation():
		# Calculate the kernel distances of examples to clusters
		# Get the nearest cluster for each example
		# Then check if the partition changed

		distances = np.array([KCM_FGH._dist_to_cluster(cluster) \
						for cluster in KCM_FGH.clusters])
		KCM_FGH.score = np.sum(distances)
		new_partition = np.argmin(distances, axis=0)
		equals = new_partition==KCM_FGH.partition
		KCM_FGH.converged = np.sum(equals)==len(equals)
		KCM_FGH.partition = new_partition

	@staticmethod
	def _kernel(xl, xk):
		# Calculate the Gaussian kernel distance of two examples

		squared_distances = (xl-xk)**2
		internal_sum_term = squared_distances*KCM_FGH.s
		sum_value = np.sum(internal_sum_term)
		return np.exp(-sum_value*0.5)

	@staticmethod
	def _single_pi_equation(r, s, h):
		# Calculate a partial value for the pi

		first_term = KCM_FGH._kernel(KCM_FGH.X[r], KCM_FGH.X[s])
		second_term = (KCM_FGH.X[r][h]-KCM_FGH.X[s][h])
		return float(first_term)*(second_term**2)

	@staticmethod
	def _update_single_pi(P_cartesian, h):
		# Sum all the partial values of a single pi

		P = P_cartesian[0]
		cartesian = P_cartesian[1]

		vals = [KCM_FGH._single_pi_equation(r, s, h) for (r, s) \
				in cartesian if r < s]

		KCM_FGH.pi_denominators[h] += np.sum(vals)/float(P)

	@staticmethod
	def _update_pi_denominators():
		# Calculate all pi's

		Ps_cartesians = [(len(cluster), list(cart(cluster, 
						cluster))) for cluster in KCM_FGH.clusters]

		KCM_FGH.pi_denominators = np.zeros(KCM_FGH.p)

		[KCM_FGH._update_single_pi(Ps_cartesians[i], h) \
		for i in range(KCM_FGH.c) for h in range(KCM_FGH.p)]

	@staticmethod
	def _update_hyperparams():
		# Function for global hyperparameters update

		KCM_FGH._update_pi_denominators()
		exponent = 1.0/KCM_FGH.p
		product_value = np.prod(KCM_FGH.pi_denominators)
		numerator = (1.0/KCM_FGH.sigma2)*(product_value**exponent)
		KCM_FGH.s = numerator/KCM_FGH.pi_denominators

	@staticmethod
	def _initialize(X):
		# Initialize all class variables
		# Then apply the algorithm initialization

		print("Initilization")

		KCM_FGH.X = X
		KCM_FGH.n = len(X)
		KCM_FGH.p = len(X[0])
		KCM_FGH.score = np.inf
		KCM_FGH.converged = False
		KCM_FGH.nb_iterations = 0
		KCM_FGH._initialize_sigma2()
		KCM_FGH.s = np.full(KCM_FGH.p, 1.0/KCM_FGH.sigma2)
		KCM_FGH._initialize_clusters()
		KCM_FGH._representation()
		KCM_FGH._allocate_in_clusters()

	@staticmethod
	def _start():
		# Main part of the algorithm, after initilization

		while KCM_FGH.converged==False and \
				KCM_FGH.nb_iterations < ConfigHelper.max_nb_iterations:

			KCM_FGH.nb_iterations+=1
			print("Iteration " + str(KCM_FGH.nb_iterations))

			KCM_FGH._update_hyperparams()
			KCM_FGH._representation()

			if KCM_FGH.converged==True:
				break

			KCM_FGH._allocate_in_clusters()

	@staticmethod
	def run(view_type, e, X):
		# Single run of the KCM-F-GH

		print(view_type.upper() + " View | Execution " + str(e))
			
		KCM_FGH._initialize(X)
		KCM_FGH._start()
			
		print("Run ended")
			
		return KCM_FGH._create_run_result(view_type, e)		