###############################################################################



class ConfigHelper():
	# Class for setting experiments configurations

	# Static for this project
	dataset_file = "test"	# Options: train | test
	nb_clusters = 7			# Options: [2, 7]

	# Configurable
	nb_executions = 3		# Options: [1, INF+]
	view_type = "full"		# Options: rgb | shape | full
	max_nb_iterations = 20	# Options: [1, INF+]