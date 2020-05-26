import yaml
from attacks.projected_gradient_descent import ProjectedGradientDescent
from attacks.carlini_wagner_l2 import CarliniWagnerL2


def create_attack(attack_config_path):
    with open(attack_config_path) as f:
        attack_params = yaml.load(f, Loader=yaml.FullLoader)

        if attack_params['attack'] == 'projected_gradient_descent':

            # deal with parameters not required for attack_mode == target
            class_fraction = 0.0
            if 'class_fraction' in attack_params.keys():
                class_fraction = attack_params['class_fraction']

            shot_fraction = 0.0
            if 'shot_fraction' in attack_params.keys():
                shot_fraction = attack_params['shot_fraction']

            # create the attack
            attack = ProjectedGradientDescent(
                norm=attack_params['norm'],
                epsilon=attack_params['epsilon'],
                num_iterations=attack_params['num_iterations'],
                epsilon_step=attack_params['epsilon_step'],
                project_step=attack_params['project_step'],
                attack_mode=attack_params['attack_mode'],
                class_fraction=class_fraction,
                shot_fraction=shot_fraction,
                clip_max=attack_params['clip_max'],
                clip_min=attack_params['clip_min']
            )
        elif attack_params['attack'] == 'carlini_wagner':
            attack = CarliniWagnerL2(
                targeted=attack_params['targeted'],
                confidence=attack_params['confidence'],
                c_lower=attack_params['c_lower'],
                c_upper=attack_params['c_upper'],
                binary_search_steps=attack_params['binary_search_steps'],
                max_iterations=attack_params['max_iterations'],
                abort_early=attack_params['abort_early'],
                box_lower=attack_params['box_lower'],
                box_upper=attack_params['box_upper'],
                optimizer_lr=attack_params['optimizer_lr'],
                init_rand=attack_params['init_rand'],
                attack_mode=attack_params['attack_mode'],
                class_fraction=attack_params['class_fraction'],
                shot_fraction=attack_params['shot_fraction'],
            )

        return attack

