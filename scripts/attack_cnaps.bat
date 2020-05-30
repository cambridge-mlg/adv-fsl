set PYTHONPATH=.
python ./learners/cnaps/src/run_cnaps.py --data_path E:/data/tf-meta-dataset/records --checkpoint_dir ./checkpoints/test --feature_adaptation film --dataset omniglot --mode attack -m learners/cnaps/models/meta-trained_meta-dataset_film.pt --shot 5 --attack_config_path scripts/configs/projected_gradient_descent.yaml
