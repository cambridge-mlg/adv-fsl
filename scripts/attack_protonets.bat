set PYTHONPATH=.
python ./learners/protonets/src/main.py ^
--dataset mini_imagenet ^
--data_path ./learners/maml/data ^
--checkpoint_dir ./checkpoints/protonets_miniimagenet_5_5 ^
--train_way 20 ^
--train_shot 5 ^
--test_way 5 ^
--test_shot 5 ^
--query 15 ^
--mode attack ^
-m ./learners/protonets/models/protonets_miniimagenet_5-way_5-shot.pt ^
--attack_config_path ./scripts/configs/carlini_wagner.yaml