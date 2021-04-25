ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=6
#python3 ./learners/fine_tune/src/fine_tune.py \
#        --data_path /scratch/etv21/meta_learning/ft_transfer_attacks/random/context/cnaps/cnaps_5_1/adv_task.pickle \
#        --checkpoint_dir /scratch/etv21/meta_learning/ft_transfer_attacks/transfer_random \
#        --log_file context_cnaps_5_1

python3 ./learners/fine_tune/src/fine_tune.py \
        --data_path /scratches/stroustrup/etv21/finetuner_transfer/aircraft/adv_task \
        --checkpoint_dir /scratches/stroustrup/etv21/finetuner_transfer/aircraft/ \
        --log_file transfer_resnet.txt \
        --attack_mode swap \
        --feature_extractor resnet \
        --pretrained_feature_extractor_path ./learners/cnaps/models/pretrained_resnet.pt.tar
