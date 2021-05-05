ulimit -n 50000
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1
#python3 ./learners/fine_tune/src/fine_tune.py \
#        --data_path /scratch/etv21/meta_learning/ft_transfer_attacks/random/context/cnaps/cnaps_5_1/adv_task.pickle \
#        --checkpoint_dir /scratch/etv21/meta_learning/ft_transfer_attacks/transfer_random \
#        --log_file context_cnaps_5_1

#python3 ./learners/fine_tune/src/fine_tune.py \
#        --data_path /scratches/stroustrup/etv21/finetuner_transfer/aircraft/adv_task \
#        --checkpoint_dir /scratches/stroustrup/etv21/finetuner_transfer/aircraft/ \
#        --dataset from_file \
#        --log_file transfer_resnet18_alt.txt \
#        --attack_mode swap \
#        --feature_adaptation no_adaptation \
#        --feature_extractor resnet18 \
#        --pretrained_feature_extractor_path learners/fine_tune/models/pretrained_resnet18_84.pt 

#python3 ./learners/fine_tune/src/fine_tune.py \
#        --data_path /scratches/stroustrup/etv21/finetuner_transfer/aircraft/adv_task \
#        --checkpoint_dir /scratches/stroustrup/etv21/finetuner_transfer/aircraft/ \
#        --dataset from_file \
#        --log_file transfer_resnet34.txt \
#        --attack_mode swap \
#        --feature_adaptation no_adaptation \
#        --feature_extractor resnet34 \
#        --pretrained_feature_extractor_path learners/fine_tune/models/pretrained_resnet34_84.pt

python3 ./learners/fine_tune/src/fine_tune.py \
        --data_path /scratches/stroustrup/etv21/finetuner_transfer/aircraft/adv_task \
        --checkpoint_dir /scratches/stroustrup/etv21/finetuner_transfer/aircraft/ \
        --dataset from_file \
        --log_file transfer_vgg.txt \
        --attack_mode swap \
        --feature_adaptation no_adaptation \
        --feature_extractor vgg11 \
        --pretrained_feature_extractor_path /scratches/stroustrup/jfb54/models/pretrained_vgg11_84.pt
