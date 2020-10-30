ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=2

python3 ./learners/maml/train.py \
	--data_path /mnt/etv21/swap_attacks/maml/maml_5_1/pgd_all_shifted_eps_0.05_mult_20/adv_task.pickle \
	--checkpoint_dir /home/etv21/debug \
	--attack_tasks 1 \
	--num_classes 5 \
	--shot 1 \
	--target_shot 1 \
	--dataset from_file \
	--model maml \
	--mode attack \
	--attack_model_path ./learners/maml/models/maml_mini_imagenet_5-way_1-shot.pt  \
	--inner_lr 0.01  \
	--swap_attack True 

#python3 ./learners/protonets/src/main.py \
#        --data_path /mnt/etv21/swap_attacks/protonets/protonets_5_1/pgd_all_shifted_eps_0.05_mult_20/adv_task.pickle \
#	--test_model_path ./learners/protonets/models/protonets_mini_imagenet_5-way_1-shot.pt  \
#        --checkpoint_dir /home/etv21/debug \
#        --dataset from_file \
#        --mode attack \
#        --swap_attack True \
#	--attack_tasks 1 \
#	--test_shot 1 --test_way 5 \
#	--query 1 \

#python3 ./learners/cnaps/src/run_cnaps.py \
#	--data_path /mnt/etv21/swap_attacks/cnaps/cnaps_5_1/pgd_all_shifted_eps_0.05_mult_20/adv_task.pickle \
#	--checkpoint_dir /home/etv21/debug \
#	--feature_adaptation film \
#	--dataset from_file \
#	--mode attack \
#	-m learners/cnaps/models/meta-trained_meta-dataset_film.pt \
#	--swap_attack True
