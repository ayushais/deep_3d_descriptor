#bin/bash
python train_model.py --model_name $1 --path_to_training_data $2 --path_to_testing_data $3 --path_to_store_models $4 
python train_model.py --model_name $1_retrain_slow_lr --path_to_training_data $2 --path_to_testing_data $3 --path_to_store_models $4 --learning_rate 0.00001 --fine_tune_model_name $4/$1_110062.ckpt  
