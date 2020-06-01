export PYTHONPATH=.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
python3 ./learners/maml/train.py \
--model maml \
--dataset omniglot \
--mode attack  \
--data_path /scratches/stroustrup/jfb54/adv-fsl \
--checkpoint_dir /scratch3/etv21/maml_omniglot_5_5 \
--num_classes 5 \
--shot 5 \
--target_shot 5 \
--gradient_steps 1 \
--iterations 60000 \
--tasks_per_batch 32 \
--inner_lr 0.4 \
--attack_tasks 10 \
--attack_config_path ./scripts/configs/carlini_wagner.yaml \
--attack_model_path ./learners/maml/models/maml_omniglot_5-way_5-shot.pt
