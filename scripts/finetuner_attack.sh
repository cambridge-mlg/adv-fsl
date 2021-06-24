ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=6

python3 ./learners/fine_tune/src/fine_tune.py \
        --checkpoint_dir /scratches/stroustrup/etv21/debug \
        --pretrained_feature_extractor_path ./learners/cnaps/models/pretrained_resnet.pt.tar \
        --feature_extractor resnet  \
        --data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \
        --dataset meta-dataset \
        --test_datasets aircraft \
        --attack_config_path /scratches/stroustrup/etv21/finetuner_attack/pgd_1.0_ac_0.2_as.yaml \
        --target_set_size_multiplier 1 \
        --query_test 10 \
        --attack_tasks 10 \
        --indep_eval True \
        --log_file grad_history.txt \
        --attack_mode swap \
        
