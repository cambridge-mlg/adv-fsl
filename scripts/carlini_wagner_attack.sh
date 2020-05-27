ulimit -n 50000
export META_DATASET_ROOT=~/Python/github.com/google-research/meta-dataset/
export PYTHONPATH=.
python3 ./learners/cnaps/src/run_cnaps.py --data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records --checkpoint_dir /scratch3/etv21/cnaps_omniglot_adapt --feature_adaptation film --dataset omniglot --mode attack -m learners/cnaps/models/meta-trained_meta-dataset_film.pt --shot 5 --way 5 --attack_config_path scripts/configs/carlini_wagner.yaml
