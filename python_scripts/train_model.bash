#bin/bash
source ~/tensorflow_12/bin/activate

python train_model.py --model_name $1 --path_to_training_data $2 --path_to_testing_data $3 --batch_size 32 --epochs 5 --learning_rate 0.0001 --eta 0.0005 --growth_rate 4

python train_model.py --model_name $1_retrain_slow_lr --path_to_training_data $2 --path_to_testing_data $3 --learning_rate 0.00001 --fine_tune_model_name $4$1_110062  --batch_size 32 --epochs 5 --eta 0.0005 --growth_rate 4


