ulimit -n 50000
export PYTHONPATH=.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
python3 ./learners/cnaps/src/run_cnaps.py \
--data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
--checkpoint_dir /scratch3/etv21/meta_learning/finetune_test2/eps_0.1_all_shifted \
--feature_adaptation film \
--dataset ilsvrc_2012 \
--mode attack \
-m learners/cnaps/models/meta-trained_meta-dataset_film.pt \
--shot 1 --way 5 \
--query_test 1 \
--attack_tasks 250 \
--attack_config_path /scratch3/etv21/meta_learning/finetune_test2/eps_0.1_all_shifted/pgd_all_shifted_eps=0.1_steps=200_r=3.0.yaml \
--indep_eval True \
--target_set_size_multiplier 13
--save_attack True

python3 ./learners/cnaps/src/run_cnaps.py \
--data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
--checkpoint_dir /scratch3/etv21/meta_learning/finetune_test2/eps_0.1_single_notarget \
--feature_adaptation film \
--dataset ilsvrc_2012 \
--mode attack \
-m learners/cnaps/models/meta-trained_meta-dataset_film.pt \
--shot 1 --way 5 \
--query_test 1 \
--attack_tasks 250 \
--attack_config_path /scratch3/etv21/meta_learning/finetune_test2/eps_0.1_single_notarget/pgd_random_no_target_eps=0.1_steps=200_r=3.0.yaml \
--indep_eval True \
--target_set_size_multiplier 13
--save_attack True

python3 ./learners/cnaps/src/run_cnaps.py \
--data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
--checkpoint_dir /scratch3/etv21/meta_learning/finetune_test2/eps_0.05_all_shifted \
--feature_adaptation film \
--dataset ilsvrc_2012 \
--mode attack \
-m learners/cnaps/models/meta-trained_meta-dataset_film.pt \
--shot 1 --way 5 \
--query_test 1 \
--attack_tasks 250 \
--attack_config_path /scratch3/etv21/meta_learning/finetune_test2/eps_0.05_all_shifted/pgd_all_shifted_eps=0.05_steps=200_r=3.0.yaml \
--indep_eval True \
--target_set_size_multiplier 13
--save_attack True

python3 ./learners/cnaps/src/run_cnaps.py \
--data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
--checkpoint_dir /scratch3/etv21/meta_learning/finetune_test2/eps_0.05_single_notarget \
--feature_adaptation film \
--dataset ilsvrc_2012 \
--mode attack \
-m learners/cnaps/models/meta-trained_meta-dataset_film.pt \
--shot 1 --way 5 \
--query_test 1 \
--attack_tasks 250 \
--attack_config_path /scratch3/etv21/meta_learning/finetune_test2/eps_0.05_single_notarget/pgd_random_no_target_eps=0.05_steps=200_r=3.0.yaml \
--indep_eval True \
--target_set_size_multiplier 13
--save_attack True


python3 ./learners/fine_tune/src/fine_tune.py \
        --data_path /scratch3/etv21/meta_learning/finetune_test2/eps_0.1_all_shifted/adv_task.pickle \
        --checkpoint_dir /scratch3/etv21/meta_learning/finetune_test2/ 
        
python3 ./learners/fine_tune/src/fine_tune.py \
        --data_path /scratch3/etv21/meta_learning/finetune_test2/eps_0.1_single_notarget/adv_task.pickle \
        --checkpoint_dir /scratch3/etv21/meta_learning/finetune_test2/ 
        
python3 ./learners/fine_tune/src/fine_tune.py \
        --data_path /scratch3/etv21/meta_learning/finetune_test2/eps_0.05_all_shifted/adv_task.pickle \
        --checkpoint_dir /scratch3/etv21/meta_learning/finetune_test2/ 
        
python3 ./learners/fine_tune/src/fine_tune.py \
        --data_path /scratch3/etv21/meta_learning/finetune_test2/eps_0.05_single_notarget/adv_task.pickle \
        --checkpoint_dir /scratch3/etv21/meta_learning/finetune_test2/ 
        
