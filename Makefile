all: IP1K IP2K

.ONESHELL:

IP1K: config/config_IP1K.py
	time=$$(date +%Y%m%d-%H%M)
	python3 script_GenerateData.py config_IP1K $$time
	python3 script_InverseProblem.py config_IP1K $$time
	python3 script_MakePlots_OneKernel.py config_IP1K $$time

IP2K: config/config_IP2K.py
	time=$$(date +%Y%m%d-%H%M)
	python script_GenerateData.py config_IP2K $$time
	python script_InverseProblem.py config_IP2K $$time
	python script_MakePlots_OneKernel.py config_IP2K $$time

Convergence: config/config_Convergence.py
	time=$$(date +%Y%m%d-%H%M)
	bash run_Convergence.sh $$time
	python script_ConvergenceAnalysis.py config_Convergence $$time

ConvergenceAll: config/config_Convergence.py
	time=$$(date +%Y%m%d-%H%M)
	bash run_ConvergenceAll.sh $$time
	python script_ConvergenceAnalysis.py config_Convergence $$time