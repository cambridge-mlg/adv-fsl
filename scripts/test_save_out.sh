ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=6
python3 ./learners/protonets/src/main.py \
--data_path /scratches/stroustrup/jfb54/miniImageNet/ \
--checkpoint_dir /scratch/etv21/debug \
--dataset mini_imagenet \
--mode attack \
--test_model_path ./learners/protonets/models/protonets_mini_imagenet_5-way_1-shot.pt  \
--test_shot 1 --test_way 5 \
--query 1 \
--attack_tasks 2 \
--attack_config_path ./scripts/configs/pgd_test.yaml \
--indep_eval True \
--target_set_size_multiplier 1 \
--save_attack True
python3 ./learners/fine_tune/src/fine_tune.py \
	--data_path /scratch/etv21/debug/adv_task.pbz2 \
	--checkpoint_dir /scratch/etv21/debug/ \
	--log_file test_load.txt \
	--feature_extractor protonets_convnet \
	--feature_adaptation no_adaptation \
	--pretrained_feature_extractor_path ./learners/protonets/models/protonets_mini_imagenet_pretrained_convnet.pt 




