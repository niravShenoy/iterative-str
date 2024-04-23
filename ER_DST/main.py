from __future__ import print_function
import sys
import os
import shutil
import time
import argparse
import logging
import hashlib
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import sparselearning
from models import cifar_resnet, initializers, vgg
from sparselearning.core import Masking, CosineDecay, LinearDecay
from sparselearning.resnet_cifar100 import ResNet34, ResNet18, ResNet50
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

models = {}
models['ResNet18'] = (ResNet18)
models['ResNet20'] = ()
models['ResNet34'] = (ResNet34)
models['ResNet50'] = (ResNet50)
models['vgg19'] = ()
models['vgg16'] = ()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("SAVING")
    torch.save(state, filename)


def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}.log'.format(args.model, args.init_density, args.final_density, args.sparse_init, args.method, args.init_method, args.prune, args.is_prune, args.seed, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


def train(args, model, device, train_loader, optimizer, epoch, mask=None):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if args.is_prune == 'True':
            if mask is not None: mask.step(copy.deepcopy(model))
            else: optimizer.step()
        if args.is_prune == 'False':
            optimizer.step()
            # to make sure mask is applied throughout training
            mask.apply_mask()
        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}% '.format(
                epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                100. * batch_idx / len(train_loader), loss.item(), correct, n, 100. * correct / float(n)))
            # mask.print_sparsity()

    # training summary
    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Training summary' ,
        train_loss/batch_idx, correct, n, 100. * correct / float(n)))

def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch GraNet for sparse training')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--batch-size-jac', type=int, default=200, metavar='N',
                        help='batch size for jac (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=1.0e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--nolr_scheduler', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--is-prune', type=str, default='False', help='Choose if pruning occurs or now.')
    parser.add_argument('--flow-preservation', type=str, default='True', help='Does flow preservation after ER init')
    parser.add_argument('--ortho-repair', type=str, default='True', help='Orthogonal Repair on Masked Init')
    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('='*80)
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))

        if args.data == 'mnist':
            train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
            num_class = 10
        elif args.data == 'cifar10':
            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split, max_threads=args.max_threads)
            num_class = 10
        elif args.data == 'cifar100':
            train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split,  max_threads=args.max_threads)
            num_class = 100
        if args.model not in models:
            print('You need to select an existing model via the --model argument. Available models include: ')
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        else:
            if args.model == 'ResNet50':
                model = ResNet50(c=num_class).to(device)
            elif args.model == 'ResNet18':
                model = ResNet18(c=num_class).to(device)
            elif args.model == 'ResNet20':
                model = cifar_resnet.Model.get_model_from_name('cifar_resnet_20', initializers.kaiming_normal, outputs=num_class).to(device)
            elif args.model == 'vgg19':
                model = vgg.VGG(depth=19, dataset=args.data, batchnorm=True).to(device)
            elif args.model == 'vgg16':
                model = vgg.VGG(depth=16, dataset=args.data, batchnorm=True).to(device)
            else:
                cls, cls_args = models[args.model]
                if args.data == 'cifar100':
                    cls_args[2] = 100
                model = cls(*(cls_args + [args.save_features, args.bench])).to(device)
                print_and_log(model)
                print_and_log('='*60)
                print_and_log(args.model)
                print_and_log('='*60)

                print_and_log('='*60)
                print_and_log('Prune mode: {0}'.format(args.prune))
                print_and_log('Growth mode: {0}'.format(args.growth))
                print_and_log('Redistribution mode: {0}'.format(args.redistribution))
                print_and_log('='*60)


        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        if args.nolr_scheduler:
            lr_scheduler = None
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier], last_epoch=-1)

        if args.resume:
            if os.path.isfile(args.resume):
                print_and_log("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint)
                original_acc = evaluate(args, model, device, test_loader)


        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale = None,
                                       dynamic_loss_scale = True,
                                       dynamic_loss_args = {'init_scale': 2 ** 16})
            model = model.half()


        mask = None
        if args.sparse:
            decay = CosineDecay(args.prune_rate, len(train_loader)*(args.epochs*args.multiplier))
            mask = Masking(optimizer, prune_rate=args.prune_rate, death_mode=args.prune, prune_rate_decay=decay, growth_mode=args.growth,
                           redistribution_mode=args.redistribution, args=args, train_loader=train_loader)
            mask.add_module(model, sparse_init=args.sparse_init)

        best_acc = 0.0
        for epoch in range(1, args.epochs*args.multiplier + 1):

            # save models
            save_path = './save/' + str(args.model) + '/' + str(args.sparse_init)  + '/' + str(args.init_method) + '/' +str(args.seed)
            save_subfolder = os.path.join(save_path, 'sparsity_' + str(args.final_density))
            if not os.path.exists(save_subfolder): os.makedirs(save_subfolder)

            t0 = time.time()
            train(args, model, device, train_loader, optimizer, epoch, mask)
            lr_scheduler.step()
            if args.valid_split > 0.0:
                val_acc = evaluate(args, model, device, test_loader)

            # target sparsity is reached
            if epoch == args.multiplier * args.final_prune_epoch+1:
                best_acc = 0.0

            if val_acc > best_acc:
                print('Saving model')
                best_acc = val_acc
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=os.path.join(save_subfolder, '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_model_final.pth'.format(args.model, args.init_density, args.final_density, args.sparse_init, args.method, args.init_method, args.prune, args.is_prune, args.seed)))

            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))

        print('Testing model')
        model.load_state_dict(torch.load(os.path.join(save_subfolder, '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_model_final.pth'.format(args.model, args.init_density, args.final_density, args.sparse_init, args.method, args.init_method, args.prune, args.is_prune, args.seed)))['state_dict'])
        test_acc = evaluate(args, model, device, test_loader, is_test_set=True)
        print('Test accuracy is:', test_acc)


if __name__ == '__main__':
   main()
