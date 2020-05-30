set PYTHONPATH=.
python ./learners/protonets_miniimagenet/attack.py ^
--checkpoint_dir ./checkpoints/proto-nets_mini-imagenet_5_5 ^
--data_path ./learners/protonets_miniimagenet/materials/ ^
--way 5 ^
--shot 5 ^
--query 5 ^
--attack_tasks 3 ^
--attack_config_path ./scripts/configs/carlini_wagner.yaml ^
--load ./learners/protonets_miniimagenet/models/proto-nets_mini-imagenet_5-way_5-shot.pth
