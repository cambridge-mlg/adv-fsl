ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
python3 ./learners/protonets/src/main.py \
        --data_path /scratches/stroustrup/jfb54/adv-fsl \
        --checkpoint_dir /scratch/etv21/backdoor/debug \
        --dataset mini_imagenet \
        --mode attack \
        --test_model_path ./learners/protonets/models/protonets_mini_imagenet_5-way_5-shot.pt  \
        --test_shot 5 --test_way 5 \
        --query 5 \
        --attack_tasks 1 \
        --attack_config_path /scratch/etv21/backdoor/backdoor.yaml \
        --indep_eval False \
        --target_set_size_multiplier 1



