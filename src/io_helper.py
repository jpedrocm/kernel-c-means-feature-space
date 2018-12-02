###############################################################################

from pandas import read_csv

from sklearn.preprocessing import LabelEncoder



class IOHelper():
	# Class for reading and storing data

	data_path = "data/"
	results_path = "results/"


	@staticmethod
	def _write_to_csv(frame, filename):
		# Writes some data to CSV
		
		file = IOHelper.results_path+filename+".csv"
		frame.to_csv(file, encoding="ascii")

	@staticmethod
	def store_best_results(view_type, suffix, results):
		# Store best acheived results so far into the results folder

		filename = view_type+"_"+suffix
		IOHelper._write_to_csv(results, filename)

	@staticmethod
	def read_view_with_classes(view_type, filename):
		# Reads a view from the dataset and returns its relevant features 
		# and classes as a 2-uple
		
		file = IOHelper.data_path+filename+".csv"
		data = read_csv(filepath_or_buffer=file, encoding="ascii",
						index_col=None, header=0, sep=",")

		classes = data["CLASS"]

		view = data.drop(columns="CLASS")

		low_var_columns = ["REGION-PIXEL-COUNT",
						   "SHORT-LINE-DENSITY-5", 
						   "SHORT-LINE-DENSITY-2"]

		if view_type=="shape":
			view = view.iloc[:, :9]
			view = view.drop(columns=low_var_columns)
		elif view_type=="rgb":
			view = view.iloc[:, 9:]
		elif view_type=="full":
			view = view.drop(columns=low_var_columns)
		else:
			raise ValueError("View type " + view_type + "not allowed")

		lb = LabelEncoder()

		return view.values, lb.fit_transform(classes.values)