all: IP1K IP2K

IP1K: config/config_IP1K.py
	time=$(date +%Y%m%d-%H%M)
	python script_GenerateData.py config_IP1K $(time)
	python script_InverseProblem.py config_IP1K $(time)
	python script_MakePlots_OneKernel.py config_IP1K $(time)

IP2K: config/config_IP2K.py
	time=$(date +%Y%m%d-%H%M)
	python script_GenerateData.py config_IP2K $(time)
	python script_InverseProblem.py config_IP2K $(time)
	python script_MakePlots_OneKernel.py config_IP2K $(time)