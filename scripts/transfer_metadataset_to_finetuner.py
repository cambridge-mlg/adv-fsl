import os

script = "#!/bin/bash\n"

script = script + "ulimit -n 50000\n"
script = script + "export PYTHONPATH=.\n"


resnet = {
    "name": "resnet", 
    "model_dir": "learners/cnaps/models/meta-trained_meta-dataset_protonets.pt", 
    "feat_ex_dir": "/learners/cnaps/models/pretrained_resnet.pt.tar"
    }
resnet18 = {
    "name": "resnet18", 
    "model_dir": "learners/cnaps/models/meta-trained_meta-dataset_protonets_resnet18_alt.pt", 
    "feat_ex_dir": "learners/fine_tune/models/pretrained_resnet18_84.pt"
    }
resnet34 = {
    "name": "resnet34", 
    "model_dir": "learners/cnaps/models/meta-trained_meta-dataset_protonets_resnet34.pt", 
    "feat_ex_dir": "learners/fine_tune/models/pretrained_resnet34_84.pt"
    }
    
vgg11 = {
    "name": "vgg11", 
    "model_dir": "learners/cnaps/models/meta-trained_meta-dataset_protonets_resnet18_alt.pt", 
    "feat_ex_dir": "learners/fine_tune/models/pretrained_vgg11_84.pt"
    }


gen_setting = resnet
gen_film_adaptation = "film"
transfer_settings = [resnet18, resnet34, vgg11]
dataset = "cifar10"

root_dir = "/home/etv21/rds/hpc-work/protonets_{}_gen_transfers".format(gen_setting["name"]) 
checkpoint_dir = os.path.join(root_dir, dataset)
model_dir = gen_setting["model_dir"]
config_path = root_dir + "/pgd_1.0_ac_0.2_as.yaml"
num_tasks = 100

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    

cmd = "python3 ./learners/cnaps/src/run_cnaps.py \\\n\
        --data_path /home/etv21/rds/hpc-work/records \\\n\
        --checkpoint_dir CHECKPOINT_DIR \\\n\
        --feature_adaptation FILM_ADAPTATION \\\n\
        --dataset meta-dataset \\\n\
        --mode attack \\\n\
        -m  MODELDIR\\\n\
        --classifier proto-nets \\\n\
        --query_test 10 \\\n\
        --attack_tasks MAXTASK \\\n\
        --attack_config_path CONFIGPATH \\\n\
        --target_set_size_multiplier 1 \\\n\
        --indep_eval True \\\n\
        --test_datasets DATASET \\\n\
        --save_attack True \\\n\
        --swap_attack True"
        
subbed_cmd = cmd.replace("CHECKPOINT_DIR", checkpoint_dir)
subbed_cmd = subbed_cmd.replace("DATASET", dataset)
subbed_cmd = subbed_cmd.replace("MODELDIR", model_dir)
subbed_cmd = subbed_cmd.replace("CONFIGPATH", config_path)
subbed_cmd = subbed_cmd.replace("MAXTASK", str(num_tasks))
subbed_cmd = subbed_cmd.replace("FILM_ADAPTATION", gen_film_adaptation)

script_name = "run.sh"
script_loc = os.path.join(checkpoint_dir, script_name)
output_file = open(script_loc, 'w')
output_file.write(script + "\n")
output_file.write(subbed_cmd)
output_file.close()
print("chmod +x {}".format(script_loc))
print("sbatch /home/etv21/rds/hpc-work/job_runner.sh {}".format(script_loc))
    

data_path = os.path.join(checkpoint_dir, "adv_task")
transfer_checkpoint_dir = os.path.join("/home/etv21/rds/hpc-work/finetune_TARGETFEAT_target", dataset) 

transfer_cmd = "python3 ./learners/fine_tune/src/fine_tune.py \\\n\
        --data_path DATA_PATH \\\n\
        --checkpoint_dir TRANSFER_CHECKPOINT_DIR \\\n\
        --dataset from_file \\\n\
        --log_file transfer_GENFEAT_to_TARGETFEAT.txt \\\n\
        --attack_mode swap \\\n\
        --feature_extractor TARGETFEAT \\\n\
        --feature_adaptation no_adaptation  \\\n\
        --pretrained_feature_extractor_path PRETRAINED_FEATURE_EXTRACTOR"

transfer_cmd = transfer_cmd.replace("GENFEAT", gen_setting["name"])
transfer_cmd = transfer_cmd.replace("DATA_PATH", data_path)

for t_set in transfer_settings:
    
    task_checkpoint_dir = transfer_checkpoint_dir.replace("TARGETFEAT", t_set["name"])
    if not os.path.exists(task_checkpoint_dir):
        os.makedirs(task_checkpoint_dir)
    task_cmd = transfer_cmd.replace("TRANSFER_CHECKPOINT_DIR", task_checkpoint_dir)
    task_cmd = task_cmd.replace("TARGETFEAT", t_set["name"])
    task_cmd = task_cmd.replace("PRETRAINED_FEATURE_EXTRACTOR", t_set["feat_ex_dir"])
    
    script_name = "transfer_from_{}.sh".format(gen_setting["name"])
    script_loc = os.path.join(task_checkpoint_dir, script_name)
    output_file = open(script_loc, 'w')
    output_file.write(script + "\n")
    output_file.write(task_cmd)
    output_file.close()
    print("chmod +x {}".format(script_loc))
    print("sbatch --dependency=after:SOMETHING /home/etv21/rds/hpc-work/torch_job_runner.sh {}".format(script_loc))
