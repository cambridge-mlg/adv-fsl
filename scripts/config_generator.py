
import os
import yaml

# Triples of ('dataset', way, shot)
large_scale = [('metadataset', -1, -1), ('omniglot', 5, 1), ('omniglot', 5, 5),('omniglot', 20, 1), ('omniglot', 20, 5)]
small_scale = [('omniglot', 5, 1), ('omniglot', 5, 5),('omniglot', 20, 1), ('omniglot', 20, 5),
               ('miniimagenet', 5, 1), ('miniimagenet', 5, 5)]

all_models = ['maml', 'protonets', 'cnaps']
all_attacks = ['pgd', 'carlini_wagner', 'elastic_net']

attack_modes = ['context', 'target']

default_attack_parameters = {
    'pgd': {
        'attack': 'projected_gradient_descent',
        'norm': 'inf',
        'epsilon': 0.3,
        'num_iterations': 10,
        'epsilon_step': 0.05,
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

def make_attack_name(attack_config):
    if attack_config['attack'] == 'projected_gradient_descent':
        return 'pgd_eps={}_eps_step={}'.format(attack_config['epsilon'],
                                               attack_config['epsilon_step'])
    elif attack_config['attack'] == 'carlini_wagner':
        return 'cw_conf={}_success={}_vary={}'.format(attack_config['confidence'],
                                                      attack_config['success_fraction'],
                                                      attack_config['vary_success_criteria'])
    elif attack_config['attack'] == 'elastic_net':
        return 'elastic_conf={}_beta={}_success={}_vary={}'.format(attack_config['confidence'],
                                                                   attack_config['beta'],
                                                                   attack_config['success_fraction'],
                                                                   attack_config['vary_success_criteria'])

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
                new_perm = lower_perm.copy() #Shallow copy should be fine
                new_perm[param_name] = value
                result.append(new_perm)
    return result

def dump_to_yaml(path, dict):
    f = open(path, "w")
    yaml.dump(dict, f, default_flow_style=False)
    f.close()

def main():
    ''''Stuff to configure:'''
    num_tasks = 10

    settings = small_scale
    if settings == small_scale:
        models = ['maml', 'protonets']
    elif settings == large_scale:
        models = ['cnaps']
    else:
        print("Error: Not sure what models to use for unknown scale setting")
        return

    attacks = all_attacks

    # Tuples with (fraction_adv_images_per_class, fraction_classes_with_adv_images)
    # Minimum num of adv images per class will be 1, so setting the first number very small gives us one per class
    attack_types = [(0.0001, 1.0), (1.0, 1.0)]

    attack_mode = 'context'

    # Leave the attack's entry as None to just use the default settings.
    # Unspecified parameters will have the default value
    # Specifying n parameters, each with v_i many values will result in
    # v_1 x v_2 .. x v_n many configurations per attack
    attack_parameters = {
        'pgd': [('epsilon', [0.1, 0.2, 0.3])],
        'carlini_wagner': [('confidence', [0.0, 2.0]), ('vary_success_criteria', [True, False])],
        'elastic_net': [('confidence', [0.0, 2.0]), ('beta', [0.01, 0.1])],
    }

    data_dir = '/scratches/stroustrup/jfb54/adv-fsl'

    output_dir = '/home/squishymage/tmp_output'#'/scratch3/etv21/'

    attack_configurations = []
    # 1. Generate the necessary attack configs:
    for attack in attacks:
        # Enumerate non-default params
        non_default_settings = enumerate_parameter_settings(attack_parameters[attack])
        for non_default_params in non_default_settings:
            for attack_type in attack_types:
                # Make copy of default config object
                attack_config = default_attack_parameters[attack].copy()
                # Configure non-default parameters:
                for param_name in non_default_params:
                    attack_config[param_name] = non_default_params[param_name]
                # Configure attack mode, shot fraction and class fraction
                attack_config['attack_mode'] = attack_mode
                attack_config['shot_fraction'] = attack_type[0]
                attack_config['class_fraction'] = attack_type[1]
                attack_configurations.append(attack_config)

    output_file = open(os.path.join(output_dir, 'run_exps.sh'), 'w')

    # 2. Generate command line
    for setting in settings:
        dataset_name = setting[0]
        way = setting[1]
        shot = setting[2]
        setting_name = '{}_{}-way_{}-shot'.format(dataset_name, way, shot)
        for model in models:
            for attack_config in attack_configurations:
                attack_name = make_attack_name(attack_config)
                exp_name = '{}_{}_{}'.format(model, setting_name, attack_name)

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
                    model_specific_params += '\t--data_set {} \\\n'.format(dataset_name)
                    model_specific_params += '\t--feature_adaptation film \\\n'
                    model_specific_params += '\t--mode attack \\\n'
                    model_specific_params += '\t--shot {} \\\n'.format(shot)
                    model_specific_params += '\t--way {} \\\n'.format(way)
                    model_specific_params += '\t-m {} '.format(model_path)
                elif model == 'maml':
                    target = '/learners/maml/train.py'
                    model_path = os.path.join(model_path, '{}_{}.pt'.format(model, setting_name))
                    model_specific_params += '\t--data_set {} \\\n'.format(dataset_name)
                    model_specific_params += '\t--model maml \\\n'
                    model_specific_params += '\t--mode attack \\\n'
                    model_specific_params += '\t--num_classes {} \\\n'.format(way)
                    model_specific_params += '\t--shot {} \\\n'.format(shot)
                    model_specific_params += '\t--target_shot {} \\\n'.format(shot)
                    model_specific_params += '\t--gradient_steps 1 \\\n'
                    model_specific_params += '\t--iterations 60000 \\\n'
                    model_specific_params += '\t--tasks_per_batch 32 \\\n'
                    model_specific_params += '\t--inner_lr 0.4  \\\n'
                    model_specific_params += '\t--attack_model_path {} '.format(model_path)

                elif model == 'protonets':
                    target = './learners/protonets_miniimagenet/attack.py'
                    model_path = os.path.join(model_path, '{}_{}.pt'.format(model, setting_name))
                    model_specific_params += '\t--shot {} \\\n'.format(shot)
                    model_specific_params += '\t--way {} \\\n'.format(way)
                    model_specific_params += '\t--query {} \\\n'.format(shot)
                    model_specific_params += '\t--load {} '.format(model_path)
                # Glue  it all together
                cmd = "python3 {} --data_path {} \\\n\t--checkpoint_dir {} \\\n\t--attack_config_path {} \\\n\t--attack_tasks {} \\\n".format(
                    target, data_dir, checkpoint_dir, attack_config_path, num_tasks)
                cmd += model_specific_params
                output_file.write(cmd + '\n')

    output_file.close()


if __name__ == "__main__":
    main()
