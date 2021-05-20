import os
import numpy as np

root_dir = "/home/etv21/rds/hpc-work/0.2_protonets_AQ_real"
datasets = ["aircraft", "cifar10", "cifar100", "cu_birds", "mnist", "mscoco", "quickdraw", "traffic_sign", "vgg_flower"]
num_dirs = 5
num_expected_tasks = 500
logfile_name = "combined_log.txt"

def print_average_accuracy(logfile, accuracies, descriptor, item):
    accuracy = np.array(accuracies).mean() * 100.0
    accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
    logfile.write('{0:} {1:}: {2:3.1f}+/-{3:2.1f}\n\n'.format(descriptor, item, accuracy, accuracy_confidence))


logfile_path = os.path.join(root_dir, logfile_name)
logfile = open(logfile_path, 'w')    

for dataset in datasets:
    accuracies = {}
    for dir_index in range(num_dirs):
        file_path = os.path.join(root_dir, dataset, str(dir_index), "dump.txt")
        acc_file = open(file_path, "r")
        while True:
            key_line = acc_file.readline().replace("\n", "")
            if not key_line:
                break
            accs_line = acc_file.readline().replace("\n", "")
            accs_line = accs_line.replace("[", "")
            accs_line = accs_line.replace("]", "")
            accs_str = accs_line.split(",")
            accs = [float(val) for val in accs_str]
            if key_line in accuracies:
                accuracies[key_line] = accuracies[key_line] + accs
            else:
                accuracies[key_line] = accs  
    for key in accuracies:
        print("{} {} {} {}".format(dataset, key, len(accuracies[key]), num_expected_tasks))
        assert len(accuracies[key]) >= num_expected_tasks
        print_average_accuracy(logfile, accuracies[key], key, dataset)
logfile.close()
            
            