import yaml
from attacks.projected_gradient_descent import ProjectedGradientDescent


def create_attack(attack_config_path):
    with open(attack_config_path) as f:
        attack_params = yaml.load(f, Loader=yaml.FullLoader)

        if attack_params['attack'] == 'projected_gradient_descent':
            attack = ProjectedGradientDescent(
                norm=attack_params['norm'],
                epsilon=attack_params['epsilon'],
                num_iterations=attack_params['num_iterations'],
                epsilon_step=attack_params['epsilon_step'],
                project_step=attack_params['project_step'],
                attack_mode=attack_params['attack_mode'],
                class_fraction=attack_params['class_fraction'],
                shot_fraction=attack_params['shot_fraction'],
                clip_max=attack_params['clip_max'],
                clip_min=-attack_params['clip_min']
            )

        return attack

