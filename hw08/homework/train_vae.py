import argparse
import logging
import os

import torch
import torchvision

from homework import vae


def train(trainer, epochs, log_interval, batch_size, n_plot, checkpoint_dir, checkpoint_period):
    for i in range(epochs):
        trainer.train(i, log_interval)
        trainer.test(i, batch_size, log_interval, n_plot)
        if (i + 1) % checkpoint_period == 0:
            trainer.save(os.path.join(checkpoint_dir, f'epoch{i}'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-root', type=str, default='../logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_vae.log')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_plot', type=int, default=8)
    parser.add_argument('--checkpoint_dir', default='ckpt_vae')
    parser.add_argument('--checkpoint_period', type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_root, args.log_name)),
            logging.StreamHandler()],
        level=logging.INFO)

    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.mnist.FashionMNIST(args.data_root, train=True, transform=transform, download=True)
    test_data = torchvision.datasets.mnist.FashionMNIST(args.data_root, train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = vae.VAE()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    trainer = vae.Trainer(model, train_loader, test_loader, optimizer, vae.loss_function, device,
                          log_dir=os.path.join(args.log_root, 'tensorboard'))
    train(trainer, args.epochs, args.log_interval, args.batch_size, args.n_plot, args.checkpoint_dir,
          args.checkpoint_period)


if __name__ == '__main__':
    main()
