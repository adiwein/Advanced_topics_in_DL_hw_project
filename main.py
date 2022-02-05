import torch
import argparse

from MoCo import MoCo
from Train import Train
from data import Data


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data = Data(args)
    train_loader, val_loader = data.get_data_loaders()
    model = MoCo(args)
    train = Train(args, model, train_loader, val_loader, device)
    train.train()
    train.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--m', default=0.9, type=float, help='m is a momentum coefficient')
    parser.add_argument('--t', default=0.07, type=float, help='t is a temperature hyper-parameter')
    parser.add_argument('--k', default=65536, type=int, help='queue size')
    parser.add_argument('--dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mlp', default=False, type=bool, help='moco2')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr', default=3e-2, type=float)
    parser.add_argument('--print_freq', default=500, type=int)
    parser.add_argument('--path', default='./final_res/lr=0.03_opt=adam_batchSize=4_m=0.9_t=0.07/mlp_final', type=str)
    parser.add_argument('--path_to_saved_dicts',
                        default='./final_res/lr=0.03_opt=adam_batchSize=4_m=0.9_t=0.07/mlp/mlp_checkpoint_00.pth.tar',
                        type=str,
                        help='path to saved checkpoint')
    parser.add_argument('--load_checkpoint', default=False, type=bool)
    parser.add_argument('--data_dir_path', default='./imagenette2/', type=str)

    parse_args = parser.parse_args()
    main(parse_args)
