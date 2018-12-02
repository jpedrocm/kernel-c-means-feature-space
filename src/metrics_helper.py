###############################################################################

from pandas import DataFrame, Series
from sklearn.metrics import adjusted_rand_score

from io_helper import IOHelper



class MetricsHelper():
	# Class for updating, calculating and storing metrics

	# Best objective score, rand index and other execution information
	metrics = DataFrame([10e9], columns=["objective_score"])
	# Best partition	
	partition = DataFrame(columns=["partition"])	
	# Hyperparams of best partition		
	hyperparams = DataFrame(columns=["hyperparams"])
	# Cluster sizes of best partition			
	sizes = None


	@staticmethod
	def _calculate_corrected_rand_score(original_classes, predicted_clusters):
		# Calculate the Corrected Rand Index of a partition

		rand_score = adjusted_rand_score(original_classes, predicted_clusters)
		MetricsHelper.metrics["rand_score"] = rand_score

	@staticmethod
	def _calculate_cluster_sizes():
		# Calculate cluster sizes for a partition

		MetricsHelper.sizes = MetricsHelper.partition["partition"].value_counts()

	@staticmethod
	def update_best_metrics(execution_results, original_classes):
		# Update the metrics based on the best objective score

		objective_score = execution_results["objective_score"]
		hyperparams = execution_results["hyperparams"]
		partition = execution_results["partition"]
		iterations = execution_results["iterations"]
		execution =  execution_results["execution"]
		view_type = execution_results["view_type"]
		converged = execution_results["converged"]

		if objective_score < MetricsHelper.metrics["objective_score"][0]:
			MetricsHelper.hyperparams["hyperparams"] = hyperparams
			MetricsHelper.partition["partition"] = partition
			MetricsHelper.metrics["objective_score"] = objective_score
			MetricsHelper.metrics["iterations"] = iterations
			MetricsHelper.metrics["execution"] = execution
			MetricsHelper.metrics["view_type"] = view_type
			MetricsHelper.metrics["converged"] = converged
			MetricsHelper._calculate_cluster_sizes()
			MetricsHelper._calculate_corrected_rand_score(partition, 
													original_classes)
			IOHelper.store_best_results(view_type, "info", 
										MetricsHelper.metrics)
			IOHelper.store_best_results(view_type, "hyperparams", 
										MetricsHelper.hyperparams)
			IOHelper.store_best_results(view_type, "partition",
										MetricsHelper.partition)
			IOHelper.store_best_results(view_type, "cluster_sizes",
										MetricsHelper.sizes)