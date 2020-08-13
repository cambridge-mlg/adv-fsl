import os
import yaml

def dump_to_yaml(path, dict):
    f = open(path, "w")
    yaml.dump(dict, f, default_flow_style=False)
    f.close()


   
def tail(f, lines=1, _buffer=4098):
    """Tail a file and get X lines from the end"""
    # place holder for the lines found
    lines_found = []

    # block counter will be multiplied by buffer
    # to get the block size from the end
    block_counter = -1

    # loop until we find X lines
    while len(lines_found) < lines:
        try:
            f.seek(block_counter * _buffer, os.SEEK_END)
        except IOError:  # either file is too small, or too many lines requested
            f.seek(0)
            lines_found = f.readlines()
            break

        lines_found = f.readlines()

        # we found enough lines, get out
        # Removed this line because it was redundant the while will catch
        # it, I left it for history
        # if len(lines_found) > lines:
        #    break

        # decrement the block counter to get the
        # next X bytes
        block_counter -= 1

    return lines_found[-lines:]

working_dir = "cnaps_5_1"
# Remember to update the model for protonets and maml

output_file = open(os.path.join(working_dir, "all_logs.txt"), 'w')


#loss = ["all_shifted", "random_no_target"]
#eps = [0.1, 0.05]
#num_steps = [20, 50, 100, 200, 500]
#eps_step_ratio = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

loss = ["all_shifted"]
eps = [0.1]
num_steps = [20, 50]
eps_step_ratio = [0.25, 0.5]


eps_step = []
names = []

             
for e in eps:
    for l in loss:
        clean_accs = []
        gen_accs = []
        eval_accs = []
        for r in eps_step_ratio:
            clean_accs.append([])
            gen_accs.append([])
            eval_accs.append([])
            
        for ri, r in enumerate(eps_step_ratio):
            for n in num_steps:
                eps_name = "pgd_{}_eps={}_steps={}_r={}".format(l, e, n, r)
                names.append(eps_name)
                input_dir = os.path.join(working_dir, eps_name)
                input_log = open(os.path.join(input_dir, "log.txt"), "r")
                # We're only interested in the last six lines, specifying dir and acc, which is enough to infer the rest.
                lines = tail(input_log, 8)
                
                # Dump the last non-empty lines into the combined log.
                # Extract the accuracy from the last 3 lines
                for line in lines:
                    if line == '\n':
                        continue
                    output_file.write(line)
                    if line.startswith("Before"):
                        acc = line.split()[-1]
                        clean_accs[ri].append(acc)
                    elif line.startswith("After"):
                        acc = line.split()[-1]
                        gen_accs[ri].append(acc)
                    elif line.startswith("Indep"):
                        acc = line.split()[-1]
                        eval_accs[ri].append(acc)
                        
        print("eps = {}, loss = {}".format(e, l))
        print(num_steps)
        for ri, r in enumerate(eps_step_ratio):
            line = "{},".format(r)
            for ni, n in enumerate(num_steps):
                line += clean_accs[ri][ni] + ","
                line += gen_accs[ri][ni] + ","
                line += eval_accs[ri][ni] + ","
            print(line)

    

