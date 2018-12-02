###############################################################################

## For reproducibility, do not change this
import random as rn
import numpy.random as rnpy

rn.seed(279)
rnpy.seed(685)
#############################################

from config_helper import ConfigHelper
from io_helper import IOHelper
from metrics_helper import MetricsHelper
from kcm_fgh import KCM_FGH



def main():

	view_type = ConfigHelper.view_type
	filename = ConfigHelper.dataset_file

	X, y = IOHelper.read_view_with_classes(view_type, filename)

	[MetricsHelper.update_best_metrics(KCM_FGH.run(view_type, e, X), y) \
	 for e in range(1, ConfigHelper.nb_executions+1)]


if __name__ == "__main__":
	main()