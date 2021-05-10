import os

script = "#!/bin/bash\n"

script = script + "ulimit -n 50000\n"
script = script + "export PYTHONPATH=.\n"


checkpoint_dir = "/home/etv21/rds/hpc-work/0.2_protonets_AQ/DATASET/TASKNUM" #/home/etv21/rds/hpc-work/large_eps_0.2_protonets_AQ
dataset = "mnist"
model_dir = "learners/cnaps/models/meta-trained_meta-dataset_protonets_film_AQ.pt"
config_path = "/home/etv21/rds/hpc-work/0.2_protonets_AQ/pgd_1.0_ac_0.2_as.yaml"

checkpoint_dir = checkpoint_dir.replace("DATASET", dataset)

cmd = "python3 ./learners/cnaps/src/run_cnaps.py \\\n\
        --data_path /home/etv21/rds/hpc-work/records \\\n\
        --checkpoint_dir CHECKPOINT_DIR \\\n\
        --feature_adaptation film \\\n\
        --dataset meta-dataset \\\n\
        --mode attack \\\n\
        -m  MODELDIR\\\n\
        --classifier proto-nets \\\n\
        --query_test 10 \\\n\
        --continue_from_task MINTASK  \\\n\
        --attack_tasks MAXTASK \\\n\
        --attack_config_path CONFIGPATH \\\n\
        --target_set_size_multiplier 1 \\\n\
        --indep_eval True \\\n\
        --test_datasets DATASET \\\n\
        --swap_attack True"
        
subbed_cmd = cmd.replace("CHECKPOINT_DIR", checkpoint_dir)
subbed_cmd = subbed_cmd.replace("DATASET", dataset)
subbed_cmd = subbed_cmd.replace("MODELDIR", model_dir)
subbed_cmd = subbed_cmd.replace("CONFIGPATH", config_path)

for i in range(0, 5):
    min_task = 100*i
    max_task = 100*(i+1)
    task_cmd = subbed_cmd.replace("MINTASK", str(min_task)).replace("MAXTASK", str(max_task)).replace("TASKNUM", str(i))
    task_checkpoint_dir = checkpoint_dir.replace("TASKNUM", str(i))
    if not os.path.exists(task_checkpoint_dir):
        os.makedirs(task_checkpoint_dir)
    script_name = "run{}.sh".format(i)
    script_loc = os.path.join(task_checkpoint_dir, script_name)
    output_file = open(script_loc, 'w')
    output_file.write(script + "\n")
    output_file.write(task_cmd)
    output_file.close()
    print("chmod +x {}".format(script_loc))
    print("sbatch /home/etv21/rds/hpc-work/job_runner.sh {}".format(script_loc))
