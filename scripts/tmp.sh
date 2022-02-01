export PYTHONPATH=.
ulimit -n 50000
#export META_DATASET_ROOT=~/Python/github.com/google-research/meta-dataset/
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

python3 ./learners/maml/train.py \
        --data_path /scratches/stroustrup/jfb54/adv-fsl \
        --checkpoint_dir /scratch/etv21/backdoor/maml/single_same \
        --attack_config_path /scratch/etv21/backdoor/single_same_backdoor.yaml \
        --attack_tasks 2 \
        --dataset mini_imagenet \
        --model maml \
        --mode attack \
        --num_classes 5 \
        --shot 5 \
        --target_shot 5 \
        --inner_lr 0.01  \
        --attack_model_path ./learners/maml/models/maml_mini_imagenet_5-way_5-shot.pt \
        --target_set_size_multiplier 1 \
        --backdoor True



#source ./scripts/ramdisk.sh

#file_p=/scratches/stroustrup/jfb54/adv-fsl
#ramdisk $file_p file_p

#python3 ./learners/cnaps/src/run_cnaps.py \
#        --data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
#        --checkpoint_dir /scratch/etv21/meta_learning/debug \
#        --feature_adaptation film \
#        --dataset ilsvrc_2012 \
#        --mode attack \
#        -m learners/cnaps/models/meta-trained_meta-dataset_film.pt \
#        --attack_tasks 100 \
#        --attack_config_path ./scripts/configs/pgd_test.yaml \
#        --indep_eval True \
#        --target_set_size_multiplier 1 \
#        --shot 5 --way 5 --query_test 5


#python3 ./learners/cnaps/src/run_cnaps.py \
#        --data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
#        --checkpoint_dir /scratch3/etv21/meta_learning/debug \
#        --feature_adaptation film \
#        --dataset ilsvrc_2012 \
#        --mode attack \
#        -m learners/cnaps/models/meta-trained_meta-dataset_film.pt \
#        --shot 5 --way 5 \
#        --query_test 5 \
#        --attack_tasks 1 \
#        --attack_config_path ./scripts/configs/pgd_test.yaml \
#        --indep_eval True \
#        --target_set_size_multiplier 13 \
#	--swap_attack True


#python3 ./learners/maml/train.py --data_path /scratches/stroustrup/jfb54/adv-fsl \
#        --checkpoint_dir /scratch3/etv21/meta_learning/debug \
#        --attack_config_path scripts/configs/pgd_test.yaml \
#        --attack_tasks 1 \
#        --dataset mini_imagenet \
#        --model maml \
#        --mode attack \
#        --num_classes 5 \
#        --shot 1 \
#        --target_shot 1 \
#        --inner_lr 0.01  \
#        --attack_model_path ./learners/maml/models/maml_mini_imagenet_5-way_1-shot.pt \
#        --indep_eval True \
#        --target_set_size_multiplier 5 \
#        --save_attack True

#python3 ./learners/maml/train.py --data_path /scratches/stroustrup/jfb54/adv-fsl \
#        --checkpoint_dir /scratch3/etv21/meta_learning/debug \
#        --attack_config_path scripts/configs/pgd_test.yaml \
#        --attack_tasks 1 \
#        --dataset mini_imagenet \
#        --model maml \
#        --mode attack \
#        --num_classes 5 \
#        --shot 1 \
#        --target_shot 1 \
#        --inner_lr 0.01  \
#        --attack_model_path ./learners/maml/models/maml_mini_imagenet_5-way_1-shot.pt \
#        --target_set_size_multiplier 5 \
#        --swap_attack True


#python3 ./learners/cnaps/src/run_cnaps.py \
#        --data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
#        --checkpoint_dir /scratch3/etv21/meta_learning/debug \
#        --feature_adaptation film \
#        --dataset ilsvrc_2012 \
#        --mode attack \
#        -m learners/cnaps/models/meta-trained_meta-dataset_film.pt \
#        --shot 1 --way 5 \
#        --query_test 1 \
#        --attack_tasks 2 \
#        --attack_config_path ./scripts/configs/pgd_test.yaml \
#        --indep_eval True \
#        --target_set_size_multiplier 13 \
#	--save_attack True

#python3 ./learners/protonets/src/main.py --data_path $file_p \
#	--checkpoint_dir /scratch3/etv21/meta_learning/debug_2dv_5shot \
#	--attack_config_path ./scripts/configs/pgd_test.yaml \
#	--attack_tasks 1 \
##	--dataset mini_imagenet \
##	--mode attack \
#	--test_shot 5 \
#	--test_way 5 \
#	--query 5 \
#	--test_model_path ./learners/protonets/models/protonets_mini_imagenet_5-way_5-shot_2D.pt \
#	--bottleneck
