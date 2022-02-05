import os
import torch
from torch import nn
import logging
import time


class Train:
    def __init__(self, args, model,  train_loader, val_loader, device: torch.device):
        self.criterion = nn.CrossEntropyLoss().cuda(device)
        self.opt = torch.optim.Adam(model.parameters(), args.lr)
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        logging.basicConfig(filename=f'{args.path}/log_file_mlp_adam_lr={args.lr}_m=0.9_t=0.07.log',
                            level=logging.INFO)

        if args.load_checkpoint:
            self.load_checkpoint()

    def train(self):
        train_start = time.time()
        self.model.train()
        for epoch in range(self.args.epochs - self.args.start_epoch):
            start_epoch = time.time()
            epoch += self.args.start_epoch
            accuracy = 0.0
            total_accuracy = 0.0
            total_loss = 0.0
            running_loss = 0.0
            count_img = 0

            print(f'epoch = {epoch}')
            for i, data in enumerate(self.train_loader):
                start_batch = time.time()
                count_img += self.args.batch_size
                x_q, x_k = data[0]

                logits, labels = self.model(x_q, x_k)
                if logits is None or labels is None:
                    continue
                loss = self.criterion(logits, labels)
                pred = torch.argmax(logits, dim=-1)
                accuracy += (labels == pred).sum()
                total_accuracy += (labels == pred).sum()

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total_loss += loss.item()
                running_loss += loss.item()

                end_batch = time.time()
                batch_time = end_batch - start_batch
                print(f'[{epoch + 1}, {i + 1:5d}]\n'
                      f'\tloss: {loss.item(): .6f}\n'
                      f'\ttotal_loss = {total_loss / (i + 1) :.6f}\n'
                      f'\taccuracy = {100 * accuracy / self.args.batch_size:.6f}\n'
                      f'\ttotal_acc = {100 * total_accuracy / count_img: .6f}\n'
                      f'\tbatch_time = {batch_time}\n')

                if i % self.args.print_freq == 0:
                    logging.info(
                        f'\nepoch = [epoch = {epoch + 1}, iter = {i + 1:5d}]'
                        f'\n    accuracy = {100 * accuracy/ self.args.batch_size:.6f}'
                        f'\n    loss = {loss.item(): .6f}'
                        f'\n    total_loss = {total_loss / (i + 1)}'
                        f'\n    total_acc = {100 * total_accuracy / count_img: .6f}'   
                        f'\n    batch_time = {batch_time}\n')
                    running_loss = 0.0
                accuracy = 0.0

            mean_accuracy = 100 * total_accuracy / count_img
            mean_loss = total_loss / count_img
            end_epoch = time.time()
            epoch_time = end_epoch - start_epoch
            print(f'\nEpoch_time = {epoch_time}\n')
            logging.info(
                         f'\n**************end of epoch: {epoch + 1}'
                         f'\n   accuracy = {total_accuracy:.6f}'
                         f'\n   mean_accuracy = {mean_accuracy: .6f}'
                         f'\n   total_loss = {total_loss:.6f}'
                         f'\n   total_mean = {mean_loss:.6f}'
                         f'\n   epoch_time = {epoch_time}\n')

            self.save_checkpoint(self.model.state_dict(),
                                 filename=f'{self.args.path}_checkpoint_{epoch:04d}.pth.tar')

        train_end = time.time()
        train_time = train_end - train_start
        logging.info(f'\ntrain_time = {train_time}\nEnd Train')
        print(f'train_time = {train_time}\nEnd Train')

    @staticmethod
    def save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    def load_checkpoint(self):
        if os.path.isfile(self.args.path_to_saved_dicts):
            print(f'loading checkpoint {self.args.path_to_saved_dicts}')

            checkpoint = torch.load(self.args.path_to_saved_dicts)
            self.model.load_state_dict(checkpoint)
        else:
            print(f'checkpoint not fount in {self.args.path_to_saved_dicts}')

    def test(self):
        start_time = time.time()
        logging.info('Start Validation')
        self.model.eval()
        count_img = 0
        accuracy = 0.0
        total_accuracy = 0.0
        running_loss = 0.0
        total_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                start_batch = time.time()
                count_img += self.args.batch_size
                x_q, x_k = data[0]

                logits, labels = self.model(x_q, x_k)
                if logits is None or labels is None:
                    continue
                loss = self.criterion(logits, labels)
                pred = torch.argmax(logits, dim=-1)
                accuracy += (labels == pred).sum()
                total_accuracy += (labels == pred).sum()

                running_loss += loss.item()
                total_loss += loss.item()

                end_batch = time.time()
                batch_time = end_batch - start_batch
                print(f'iter = [{i + 1:5d}]\n'
                      f'\tloss: {loss.item(): .6f}\n'
                      f'\ttotal_loss = {total_loss / (i + 1) :.6f}\n'
                      f'\taccuracy = {100 * accuracy / self.args.batch_size:.6f}\n'
                      f'\ttotal_acc = {100 * total_accuracy / count_img: .6f}\n'
                      f'\tbatch_time = {batch_time}\n')

                if i % self.args.print_freq == 0:
                    logging.info(
                        f'\niter = [{i + 1:5d}]'
                        f'\n    accuracy = {100 * accuracy / self.args.batch_size:.6f}'
                        f'\n    loss = {loss.item(): .6f}'
                        f'\n    running_loss = {running_loss / self.args.print_freq:.6f}'
                        f'\n    total_loss = {total_loss / (i + 1)}'
                        f'\n    batch_time = {end_batch - start_batch}')
                    running_loss = 0.0

                accuracy = 0.0

        end_time = time.time()
        total_time = end_time - start_time
        mean_accuracy = 100 * total_accuracy / count_img
        mean_loss = total_loss / count_img
        print(f'\ntotal_time = {total_time}\n')
        logging.info(
            f'\nvalidation results:'
            f'\n   accuracy = {total_accuracy:.6f}'
            f'\n   mean_accuracy = {mean_accuracy: .6f}'
            f'\n   total_loss = {total_loss:.6f}'
            f'\n   total_mean = {mean_loss:.6f}'
            f'\n   total_time = {total_time}\n')
