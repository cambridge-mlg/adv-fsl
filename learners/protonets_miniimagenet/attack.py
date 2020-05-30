import argparse

import torch
from torch.utils.data import DataLoader
import os

from learners.protonets_miniimagenet.mini_imagenet import MiniImageNet
from learners.protonets_miniimagenet.samplers import CategoriesSampler
from learners.protonets_miniimagenet.convnet import Convnet
from learners.protonets_miniimagenet.utils import pprint, set_gpu, count_acc, Averager, euclidean_metric
from attacks.attack_helpers import create_attack
from learners.maml.src.utils import save_image


class ModelWrapper:
    def __init__(self, model, way, shot, query):
        self.model = model
        self.shot = shot
        self.way = way
        self.query = query

    def get_logits(self, context_images, _unused, target_images):
        del _unused  # labels are not needed as the data is in class order
        x = model(context_images)
        x = x.reshape(self.shot, self.way, -1).mean(dim=0)
        p = x

        return euclidean_metric(model(target_images), p)

    def compute_accuracy(self, context_images, target_images):
        logits = self.get_logits(context_images, None, target_images)
        label = torch.arange(self.way).repeat(self.query)
        label = label.type(torch.cuda.LongTensor)
        acc = count_acc(logits, label)
        return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--data_path', default='./learners/protonets_miniimagenet/materials')
    parser.add_argument('--load')
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--attack_tasks', type=int, default=10, help='Number of attack tasks.')
    parser.add_argument("--attack_config_path", help="Path to attack config file in yaml format.")
    parser.add_argument("--checkpoint_dir", help="Path to checkpoint directory.")
    args = parser.parse_args()
    pprint(vars(args))

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    set_gpu(args.gpu)

    dataset = MiniImageNet('test', args.data_path)
    sampler = CategoriesSampler(dataset.label, args.attack_tasks, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=8, pin_memory=True)

    model = Convnet().cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()

    ave_acc = Averager()

    model_wrapper = ModelWrapper(model, args.way, args.shot, args.query)
    attack = create_attack(args.attack_config_path)

    for i, task in enumerate(loader, 1):
        data, _ = [_.cuda() for _ in task]
        k = args.way * args.shot
        context_images, target_images = data[:k], data[k:]
        device = data.device
        context_labels = torch.arange(args.way).repeat(args.shot)

        if attack.get_attack_mode() == 'context':
            adv_context_images, adv_context_indices = attack.generate(
                context_images,
                context_labels,
                target_images,
                model,
                model_wrapper.get_logits,
                device)

            for index in adv_context_indices:
                save_image(adv_context_images[index].cpu().detach().numpy(),
                           os.path.join(args.checkpoint_dir, 'adv_task_{}_index_{}.png'.format(i, index)))
                save_image(context_images[index].cpu().detach().numpy(),
                           os.path.join(args.checkpoint_dir, 'in_task_{}_index_{}.png'.format(i, index)))

            with torch.no_grad():
                acc_after = model_wrapper.compute_accuracy(adv_context_images, target_images)

        else:  # target
            adv_target_images = attack.generate(context_images, context_labels, target_images, model,
                                                model_wrapper.get_logits, device)
            for i in range(len(target_images)):
                save_image(adv_target_images[i].cpu().detach().numpy(),
                           os.path.join(args.checkpoint_dir, 'adv_task_{}_index_{}.png'.format(i, i)))
                save_image(target_images[i].cpu().detach().numpy(),
                           os.path.join(args.checkpoint_dir, 'in_task_{}_index_{}.png'.format(i, i)))

            with torch.no_grad():
                acc_after = model_wrapper.compute_accuracy(context_images, adv_target_images)

        acc_before = model_wrapper.compute_accuracy(context_images, target_images)

        diff = acc_before - acc_after
        print("Task = {}, Diff = {}".format(i, diff))



