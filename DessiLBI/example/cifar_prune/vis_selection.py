import matplotlib.pyplot as plt
import argparse
import numpy as np
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--txtname', type=str, default='./txtrecorder/finalvgg16bn_lbi_weight_mu500_norm_0.0003.txt', help="name \
                        of txt") 
args = parser.parse_args()

def get_info(file):
    whole_list = []
    select_list = []
    for line in file.readlines():
        if 'number of selected weight:' in line:
            select_list.append(float(line.replace('number of selected weight:', '')))
        elif 'number of weight:' in line:
            whole_list.append(float(line.replace('number of weight: ', '')))
        else:
            pass
    return whole_list, select_list
vgg_file = open(args.txtname)
whole_list, select_list = get_info(vgg_file)
print(whole_list)
print(select_list)
plt.figure()
plt.plot(range(len(whole_list)), whole_list, label='Dense Model')
plt.plot(range(len(select_list)), select_list, label = 'Sparse Model')
plt.legend()
plt.title('VGG 16 Structure')
plt.xlabel('Layer Index')
plt.ylabel('Number of Filters')
plt.savefig('vgg16.pdf')


vgg_file = open(args.txtname)
whole_list, select_list = get_info(vgg_file)
plt.figure()
plt.plot(range(len(whole_list)), np.array(select_list)/ np.array(whole_list), label='Ratio')
plt.legend()
plt.title('VGG 16 Ratio')
plt.xlabel('Layer Index')
plt.ylabel('Selection Ratio')
plt.savefig('vgg16r.pdf')

