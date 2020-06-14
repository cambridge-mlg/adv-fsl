import os
import yaml
import argparse

# Triples of ('dataset', way, shot)
large_scale = [('metadataset', -1, -1), ('omniglot', 5, 1), ('omniglot', 5, 5), ('omniglot', 20, 1),
               ('omniglot', 20, 5)]
small_scale = [('omniglot', 5, 1), ('omniglot', 5, 5), ('omniglot', 20, 1), ('omniglot', 20, 5),
               ('mini_imagenet', 5, 1), ('mini_imagenet', 5, 5)]  # ('ilsvrc_2012', 5, 1), ('ilsvrc_2012', 5, 5) for cnaps

all_models = ['maml', 'protonets', 'cnaps']

all_attacks = ['pgd', 'carlini_wagner', 'elastic_net']

attack_modes = ['context', 'target']

default_attack_parameters = {
    'pgd': {
        'attack': 'projected_gradient_descent',
        'norm': 'inf',
        'epsilon': 8.0 / 255.0,
        'num_iterations': 20,
        'epsilon_step': 2.0 / 255.0,
        'project_step': True,
        'attack_mode': None,
        'class_fraction': None,
        'shot_fraction': None,
    },
    'carlini_wagner': {
        'attack': 'carlini_wagner',
        'targeted': False,
        'confidence': 0.0,
        'c_lower': 1.0,
        'c_upper': 1.0e+10,
        'binary_search_steps': 15,
        'max_iterations': 100,
        'abort_early': True,
        'optimizer_lr': 1.0e-2,
        'init_rand': False,
        'success_fraction': 0.5,
        'vary_success_criteria': False,
        'attack_mode': None,
        'class_fraction': None,
        'shot_fraction': None,
    },
    'elastic_net': {
        'attack': 'elastic_net',
        'confidence': 0.0,
        'targeted': False,
        'learning_rate': 1.0e-2,
        'max_iterations': 500,
        'binary_search_steps': 15,
        'abort_early': True,
        'beta': 1.0e-2,
        'decision_rule': 'EN',
        'c_lower': 1.0e-3,
        'c_upper': 1.0e+10,
        'success_fraction': 0.5,
        'attack_mode': None,
        'class_fraction': None,
        'shot_fraction': None,
    }
}

default_maml_parameters = {
    'omniglot_5-way_1-shot': {
        'inner_lr': 0.4
    },
    'omniglot_5-way_5-shot': {
        'inner_lr': 0.4
    },
    'omniglot_20-way_1-shot': {
        'inner_lr': 0.1
    },
    'omniglot_20-way_5-shot': {
        'inner_lr': 0.1
    },
    'mini_imagenet_5-way_1-shot': {
        'inner_lr': 0.01
    },
    'mini_imagenet_5-way_5-shot': {
        'inner_lr': 0.01
    }
}

default_protonets_parameters = {
    'omniglot_5-way_1-shot': {
        'query': 5,
    },
    'omniglot_5-way_5-shot': {
        'query': 5,
    },
    'omniglot_20-way_1-shot': {
        'query': 5,
    },
    'omniglot_20-way_5-shot': {
        'query': 5,
    },
    'mini_imagenet_5-way_1-shot': {
        'query': 15,
    },
    'mini_imagenet_5-way_5-shot': {
        'query': 15,
    }
}

default_cnaps_parameters = {
    'omniglot_5-way_1-shot': {
        'query_test': 1
    },
    'omniglot_5-way_5-shot': {
        'query_test': 5
    },
    'omniglot_20-way_1-shot': {
        'query_test': 1
    },
    'omniglot_20-way_5-shot': {
        'query_test': 5
    },
    'meta_dataset': {
        'query_test': 10
    },
    'ilsvrc_2012_5-way_1-shot': {
        'query_test': 1
    },
    'ilsvrc_2012_5-way_5-shot': {
        'query_test': 5
    },
    'ilsvrc_2012_20-way_1-shot': {
        'query_test': 1
    },
    'ilsvrc_2012_20-way_5-shot': {
        'query_test': 5
    }
}


def make_attack_name(attack_config):
    class_shot_fraction_str = 'cf={:.2f}_sf={:.2f}'.format(attack_config['class_fraction'],
                                                           attack_config['shot_fraction'], )
    if attack_config['attack'] == 'projected_gradient_descent':
        return 'pgd_{}_eps={:.2f}_eps_step={:.3f}'.format(class_shot_fraction_str,
                                                          attack_config['epsilon'],
                                                          attack_config['epsilon_step'])
    elif attack_config['attack'] == 'carlini_wagner':
        return 'cw_{}_conf={:.2f}_success={}_vary={}'.format(class_shot_fraction_str,
                                                             attack_config['confidence'],
                                                             attack_config['success_fraction'],
                                                             attack_config['vary_success_criteria'])
    elif attack_config['attack'] == 'elastic_net':
        return 'elastic_{}_conf={:.2f}_beta={:.2f}_success={}'.format(class_shot_fraction_str,
                                                                      attack_config['confidence'],
                                                                      attack_config['beta'],
                                                                      attack_config['success_fraction'])


def enumerate_parameter_settings(parameters):
    if len(parameters) == 0:
        return []

    param_name = parameters[0][0]
    param_vals = parameters[0][1]
    result = []
    if len(parameters) == 1:
        for value in param_vals:
            result.append({param_name: value})
    else:
        lower_perms = enumerate_parameter_settings(parameters[1:])
        for value in param_vals:
            for lower_perm in lower_perms:
                new_perm = lower_perm.copy()  # Shallow copy should be fine
                new_perm[param_name] = value
                result.append(new_perm)
    return result


def dump_to_yaml(path, dict):
    f = open(path, "w")
    yaml.dump(dict, f, default_flow_style=False)
    f.close()


def main(gpu_num, output_dir, num_tasks, script_name):
    ''''Stuff to configure:'''

    exp_settings = [{'model': 'maml', 'problem': ('omniglot', 5, 1)},
                    {'model': 'maml', 'problem': ('omniglot', 5, 5)},
                    {'model': 'maml', 'problem': ('mini_imagenet', 5, 1)},
                    {'model': 'maml', 'problem': ('mini_imagenet', 5, 5)},
                    {'model': 'protonets', 'problem': ('omniglot', 5, 1)},
                    {'model': 'protonets', 'problem': ('omniglot', 5, 5)},
                    {'model': 'protonets', 'problem': ('mini_imagenet', 5, 1)},
                    {'model': 'protonets', 'problem': ('mini_imagenet', 5, 5)},
                    {'model': 'cnaps', 'problem': ('omniglot', 5, 1)},
                    {'model': 'cnaps', 'problem': ('omniglot', 5, 5)},
                    {'model': 'cnaps', 'problem': ('ilsvrc_2012', 5, 1)},
                    {'model': 'cnaps', 'problem': ('ilsvrc_2012', 5, 5)},
                    ]

    attacks = ['pgd']

    # Tuples with (fraction_adv_images_per_class, fraction_classes_with_adv_images)
    # Minimum num of adv images per class will be 1, so setting the first number very small gives us one per class
    attack_types = [(0.0001, 1.0), (1.0, 0.5)]

    attack_mode = 'context'

    # Leave the attack's entry as None to just use the default settings.
    # Unspecified parameters will have the default value
    # Specifying n parameters, each with v_i many values will result in
    # v_1 x v_2 .. x v_n many configurations per attack
    attack_parameters = {
        'carlini_wagner': [('vary_success_criteria', [True, False])],
    }

    root_data_dir = '/scratches/stroustrup/jfb54/'

    attack_configurations = []
    # 1. Generate the necessary attack configs:
    for attack in attacks:

        # Enumerate non-default params, if any
        # Generate this outside the loop, no need to re-generate for each attack_type
        if attack not in attack_parameters:
            # Only default parameters, list contains only an empty dictionary
            settings_list = [{}]
        else:
            settings_list = enumerate_parameter_settings(attack_parameters[attack])

        for params in settings_list:
            for attack_type in attack_types:
                # Make copy of default config object
                attack_config = default_attack_parameters[attack].copy()
                # Configure non-default parameters:
                for param_name in params:
                    attack_config[param_name] = params[param_name]
                # Configure attack mode, shot fraction and class fraction
                attack_config['attack_mode'] = attack_mode
                attack_config['shot_fraction'] = attack_type[0]
                attack_config['class_fraction'] = attack_type[1]
                attack_configurations.append(attack_config)

    output_file = open(os.path.join(output_dir, script_name), 'w')
    output_file.write('ulimit -n 50000\n')
    output_file.write('export PYTHONPATH=.\n')
    output_file.write('export CUDA_VISIBLE_DEVICES={}\n'.format(gpu_num))

    # 2. Generate command line
    for setting in exp_settings:
        model = setting['model']
        dataset_name = setting['problem'][0]
        way = setting['problem'][1]
        shot = setting['problem'][2]
        if dataset_name == 'meta_dataset':
            setting_name = dataset_name
        else:
            setting_name = '{}_{}-way_{}-shot'.format(dataset_name, way, shot)

        for attack_config in attack_configurations:
            attack_name = make_attack_name(attack_config)

            # Make checkpoint dir
            model_path = os.path.join('./learners', model, 'models')
            checkpoint_dir = os.path.join(output_dir, model, setting_name, attack_name)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            # Save config in relevant folder
            attack_config_path = os.path.join(checkpoint_dir, '{}.yaml'.format(attack_name))
            dump_to_yaml(attack_config_path, attack_config)

            # Generate command
            model_specific_params = ''
            if model == 'cnaps':
                target = './learners/cnaps/src/run_cnaps.py'
                model_path = os.path.join(model_path, 'meta-trained_meta-dataset_film.pt')
                data_dir = os.path.join(root_data_dir, 'tf-meta-dataset/records')
                model_specific_params += '\t--dataset {} \\\n'.format(dataset_name)
                model_specific_params += '\t--feature_adaptation film \\\n'
                model_specific_params += '\t--mode attack \\\n'
                model_specific_params += '\t--shot {} \\\n'.format(shot)
                model_specific_params += '\t--way {} \\\n'.format(way)
                model_specific_params += '\t--query_test {} \\\n'.format(
                    default_cnaps_parameters[setting_name]['query_test'])
                model_specific_params += '\t-m {} '.format(model_path)
            elif model == 'maml':
                target = './learners/maml/train.py'
                model_path = os.path.join(model_path, '{}_{}.pt'.format(model, setting_name))
                data_dir = os.path.join(root_data_dir, 'adv-fsl')
                model_specific_params += '\t--dataset {} \\\n'.format(dataset_name)
                model_specific_params += '\t--model maml \\\n'
                model_specific_params += '\t--mode attack \\\n'
                model_specific_params += '\t--num_classes {} \\\n'.format(way)
                model_specific_params += '\t--shot {} \\\n'.format(shot)
                model_specific_params += '\t--inner_lr {}  \\\n'.format(
                    default_maml_parameters[setting_name]['inner_lr'])
                model_specific_params += '\t--attack_model_path {} '.format(model_path)

            elif model == 'protonets':
                target = './learners/protonets/src/main.py'
                model_path = os.path.join(model_path, '{}_{}.pt'.format(model, setting_name))
                data_dir = os.path.join(root_data_dir, 'adv-fsl')
                model_specific_params += '\t--dataset {} \\\n'.format(dataset_name)
                model_specific_params += '\t--mode attack \\\n'
                model_specific_params += '\t--test_shot {} \\\n'.format(shot)
                model_specific_params += '\t--test_way {} \\\n'.format(way)
                model_specific_params += '\t--query {} \\\n'.format(
                    default_protonets_parameters[setting_name]['query'])
                model_specific_params += '\t--test_model_path {} '.format(model_path)
            # Glue  it all together
            cmd = "python3 {} --data_path {} \\\n\t--checkpoint_dir {} \\\n\t--attack_config_path {} \\\n\t--attack_tasks {} \\\n".format(
                target, data_dir, checkpoint_dir, attack_config_path, num_tasks)
            cmd += model_specific_params
            output_file.write(cmd + '\n')

    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-d", default='/scratch/etv21/meta_learning',
                        help="Directory to which adv samples, log files, etc will be saved to")
    parser.add_argument("--script_name", "-o", default='run_exps.sh',
                        help="Name of generated bash file")
    parser.add_argument("--gpu_num", "-g", type=int, default=1, help="GPU to use")
    parser.add_argument("--num_tasks", "-t", type=int, default=100, help="How many tasks per experiment")

    args = parser.parse_args()

    main(args.gpu_num, args.output_dir, args.num_tasks, args.script_name)