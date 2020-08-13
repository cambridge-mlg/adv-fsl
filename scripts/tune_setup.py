import os
import yaml

def dump_to_yaml(path, dict):
    f = open(path, "w")
    yaml.dump(dict, f, default_flow_style=False)
    f.close()

output_dir = "protonets_5_10"
gpu_num = 2
mult = 6
shot = 10
# Remember to update the model for protonets and maml

output_file = open(os.path.join(output_dir, "run.sh"), 'w')
output_file.write('ulimit -n 50000\n')
output_file.write('export PYTHONPATH=.\n')
#output_file.write('export CUDA_DEVICE_ORDER=PCI_BUS_ID\n')
output_file.write('export CUDA_VISIBLE_DEVICES={}\n'.format(gpu_num))
        
'''
cmd = "python3 ./learners/cnaps/src/run_cnaps.py \\\n"
cmd += "--data_path /scratches/stroustrup/jfb54/tf-meta-dataset/records \\\n"
cmd += "--checkpoint_dir /scratch3/etv21/meta_learning/tune/cnaps_5_{}".format(shot) + "/{} \\\n"
cmd += "--feature_adaptation film \\\n"
cmd += "--dataset ilsvrc_2012 \\\n"
cmd += "--mode attack \\\n"
cmd += "-m learners/cnaps/models/meta-trained_meta-dataset_film.pt \\\n"
cmd += "--shot {} --way 5 \\\n".format(shot)
cmd += "--query_test {} \\\n".format(shot)
cmd += "--attack_tasks 100 \\\n"
cmd += "--attack_config_path /scratch3/etv21/meta_learning/tune/cnaps_5_{}".format(shot) + "/{}/{}.yaml \\\n"
cmd += "--indep_eval True \\\n"
cmd += "--target_set_size_multiplier {}\n".format(mult)
'''

cmd = "python3 ./learners/protonets/src/main.py \\\n"
cmd += "--data_path /home/jfb54/data \\\n"
cmd += "--checkpoint_dir /home/etv21/tune/protonets_5_{}".format(shot) + "/{} \\\n"
cmd += "--dataset mini_imagenet \\\n"
cmd += "--mode attack \\\n"
cmd += "--test_model_path ./learners/protonets/models/protonets_mini_imagenet_5-way_5-shot.pt  \\\n"
cmd += "--test_shot {} --test_way 5 \\\n".format(shot)
cmd += "--query {} \\\n".format(shot)
cmd += "--attack_tasks 100 \\\n"
cmd += "--attack_config_path /home/etv21/tune/protonets_5_{}".format(shot) + "/{}/{}.yaml \\\n"
cmd += "--indep_eval True \\\n"
cmd += "--target_set_size_multiplier {}\n".format(mult)


params = {
    'attack': 'projected_gradient_descent',
    'norm': 'inf',
    'epsilon': 8.0 / 255.0,
    'num_iterations': 20,
    'epsilon_step': 2.0 / 255.0,
    'project_step': True,
    'attack_mode': 'context',
    'class_fraction': 1.0,
    'shot_fraction': 1.0,
    'use_true_target_labels': True,
    'target_loss_mode': 'all',
    'targeted': True,
    'targeted_labels': 'shifted',
}

loss = ["all_shifted", "random_no_target"]
eps = [0.1, 0.05]
num_steps = [20, 50, 100, 200, 500]
eps_step_ratio = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

eps_step = []
names = []
             
for l in loss:
    if l == 'all_shifted':
        params['target_loss_mode'] = 'all'
        params['targeted'] = True
        params['targeted_labels'] = 'shifted'
    else:
        params['target_loss_mode'] = 'random'
        params['targeted'] = False
    
    for e in eps:
        for n in num_steps:
            for r in eps_step_ratio:
                eps_step = (e/n)*r
                params['epsilon'] = e
                params['epsilon_step'] = eps_step
                params['num_iterations'] = n
                eps_name = "pgd_{}_eps={}_steps={}_r={}".format(l, e, n, r)
                names.append(eps_name)
                if not os.path.exists(eps_name):
                    os.makedirs(os.path.join(output_dir, eps_name))
                
                attack_config_path = os.path.join(output_dir, eps_name, '{}.yaml'.format(eps_name))
                dump_to_yaml(attack_config_path, params)
                output_file.write(cmd.format(eps_name, eps_name, eps_name) + '\n')


    
   

    
