# data path should have rundata under a subdir 
# eg: 
#   ~/Desktop/sim_data is tha parameter
#   but run data are under ~/Desktop/sim_data/run1 , ~/Desktop/sim_data/run2 etc
# 

python model.py --model comma.ai  --datapath ~/Desktop/sim_data/ --epochs 20  --drive
# python model.py --model nvidia  --drive --datapath ~/Desktop/sim_data/
# python model.py --model comma.ai   --datapath comma.ai/data/ --balanced --epochs 2
