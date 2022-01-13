import yaml
import numpy as np
import random
from attacks.projected_gradient_descent import ProjectedGradientDescent
from attacks.carlini_wagner_l2 import CarliniWagnerL2
from attacks.elastic_net import ElasticNet
from attacks.shift_attack import ShiftAttack
from attacks.uap import UapAttack
from attacks.random import RandomAttack


def create_attack(attack_config_path, checkpoint_dir):
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

            randomize_attack_params=False
            if 'randomize' in attack_params.keys():
                randomize_attack_params = attack_params['randomize']
            shuffle_context = False
            if 'shuffle_context' in attack_params.keys():
                shuffle_context = attack_params['shuffle_context']

            shuffle_context_mode = 'none'
            if 'shuffle_context_mode' in attack_params.keys():
                shuffle_context_mode = attack_params['shuffle_context_mode']
                
            sub_context_size_coeff = 1.0
            if 'sub_context_size_coeff' in attack_params.keys():
                sub_context_size_coeff = attack_params['sub_context_size_coeff']

            hot_start = False
            hot_start_location = ""
            if "hot_start" in attack_params.keys():
                hot_start = True
            assert "hot_start_location" in attack_params.keys()
            hot_start_location = attack_params["hot_start_location"]

            # create the attack
            if randomize_attack_params:
                epsilon = np.random.uniform(0.025, 0.055)
                num_iterations = np.random.randint(low=5, high=51)
                epsilon_step = epsilon * 3.0 / float(num_iterations)
                # print("eps={}, iters={}, step={}".format(epsilon, num_iterations, epsilon_step))
            else:
                epsilon = attack_params['epsilon']
                num_iterations = attack_params['num_iterations']
                epsilon_step = attack_params['epsilon_step']

            attack = ProjectedGradientDescent(
                checkpoint_dir=checkpoint_dir,
                norm=attack_params['norm'],
                epsilon=epsilon,
                num_iterations=num_iterations,
                epsilon_step=epsilon_step,
                project_step=attack_params['project_step'],
                attack_mode=attack_params['attack_mode'],
                class_fraction=class_fraction,
                shot_fraction=shot_fraction,
                use_true_target_labels=attack_params['use_true_target_labels'],
                target_loss_mode=attack_params['target_loss_mode'],
                targeted=attack_params['targeted'],
                targeted_labels=attack_params['targeted_labels'],
                shuffle_context=shuffle_context,
                shuffle_context_mode=shuffle_context_mode,
                sub_context_size_coeff=sub_context_size_coeff,
                hot_start=hot_start,
                hot_start_location=hot_start_location
            )
        elif attack_params['attack'] == 'carlini_wagner':
            attack = CarliniWagnerL2(
                checkpoint_dir=checkpoint_dir,
                targeted=attack_params['targeted'],
                confidence=attack_params['confidence'],
                c_lower=attack_params['c_lower'],
                c_upper=attack_params['c_upper'],
                binary_search_steps=attack_params['binary_search_steps'],
                max_iterations=attack_params['max_iterations'],
                abort_early=attack_params['abort_early'],
                optimizer_lr=attack_params['optimizer_lr'],
                init_rand=attack_params['init_rand'],
                attack_mode=attack_params['attack_mode'],
                success_fraction=attack_params['success_fraction'],
                vary_success_criteria=attack_params['vary_success_criteria'],
                class_fraction=attack_params['class_fraction'],
                shot_fraction=attack_params['shot_fraction'],
                use_true_target_labels=attack_params['use_true_target_labels']
            )
        elif attack_params['attack'] == 'elastic_net':
            attack = ElasticNet(
                checkpoint_dir=checkpoint_dir,
                confidence=attack_params['confidence'],
                targeted=attack_params['targeted'],
                learning_rate=attack_params['learning_rate'],
                max_iterations=attack_params['max_iterations'],
                binary_search_steps=attack_params['binary_search_steps'],
                abort_early=attack_params['abort_early'],
                beta=attack_params['beta'],
                decision_rule=attack_params['decision_rule'],
                c_lower=attack_params['c_lower'],
                c_upper=attack_params['c_upper'],
                success_fraction=attack_params['success_fraction'],
                attack_mode=attack_params['attack_mode'],
                class_fraction=attack_params['class_fraction'],
                shot_fraction=attack_params['shot_fraction'],
                use_true_target_labels=attack_params['use_true_target_labels']
            )
        elif attack_params['attack'] == 'shift':
            attack = ShiftAttack(
                checkpoint_dir=checkpoint_dir,
                attack_mode=attack_params['attack_mode'],
                class_fraction=attack_params['class_fraction'],
                shot_fraction=attack_params['shot_fraction']
            )
        elif attack_params['attack'] == "uap":
            attack = UapAttack(
                checkpoint_dir=checkpoint_dir,
                attack_mode=attack_params['attack_mode'],
                class_fraction=attack_params['class_fraction'],
                shot_fraction=attack_params['shot_fraction'],
                perturbation_image_path=attack_params['perturbation_image_path']
            )
        elif attack_params['attack'] == "random":
            attack = RandomAttack(
                checkpoint_dir=checkpoint_dir,
                attack_mode=attack_params['attack_mode'],
                class_fraction=attack_params['class_fraction'],
                shot_fraction=attack_params['shot_fraction'],
                epsilon=attack_params['epsilon']
            )

        return attack

