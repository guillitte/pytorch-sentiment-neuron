import argparse

import torch

parser = argparse.ArgumentParser(description='convert_to_cpu.py')

parser.add_argument('-load_model', default='',
                    help="""Model filename to load and convert.""")

opt = parser.parse_args()

checkpoint = torch.load(opt.load_model)
embed = checkpoint['embed']
rnn = checkpoint['rnn']

checkpoint = {
    'rnn': rnn.cpu(),
    'embed': embed.cpu(),
    'opt': checkpoint['opt'],
    'epoch': checkpoint['epoch']
}
save_file = opt.load_model + '.cpu'
print('Saving to ' + save_file)
torch.save(checkpoint, save_file)
