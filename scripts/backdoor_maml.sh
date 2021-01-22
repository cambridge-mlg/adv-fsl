export PYTHONPATH=.
ulimit -n 50000
export CUDA_VISIBLE_DEVICES=0

python3 ./learners/maml/train.py \
	--data_path /scratches/stroustrup/jfb54/adv-fsl \
        --checkpoint_dir . \
        --attack_config_path ./scripts/configs/single_other_match_backdoor.yaml \
        --attack_tasks 20 \
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

#python3 ./learners/maml/train.py \
#	--data_path /scratches/stroustrup/jfb54/adv-fsl \
 #       --checkpoint_dir /scratch/etv21/backdoor/maml/all_other_match \
  #      --attack_config_path /scratch/etv21/backdoor/all_other_match_backdoor.yaml \
   #     --attack_tasks 500 \
    #    --dataset mini_imagenet \
     #   --model maml \
      #  --mode attack \
       # --num_classes 5 \
       # --shot 5 \
       # --target_shot 5 \
       # --inner_lr 0.01  \
       # --attack_model_path ./learners/maml/models/maml_mini_imagenet_5-way_5-shot.pt \
       # --target_set_size_multiplier 1 \
       # --backdoor True

