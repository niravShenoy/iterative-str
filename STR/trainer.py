import time
import torch
import tqdm
import math

from utils.eval_utils import accuracy
from utils.conv_type import STRConv, STRConvER
from utils.logging import AverageMeter, ProgressMeter


__all__ = ["train", "validate"]

def magnitude_death(mask, weight, prune_rate, num_nonzeros, num_zeros):
    num_remove = math.ceil(prune_rate * num_nonzeros)
    num_retain = math.ceil(num_nonzeros - num_remove)  # [(1 - prune_rate) * num_nonzeros]

    if num_remove == 0.0:
        return weight.data != 0.0
    
    k = math.ceil(num_zeros + num_remove)

    x1, _ = torch.sort(torch.abs(weight.data[mask == 1.0]), descending=True) # Alternate logic to extract the top [(1 - prune_rate) * num_nonzeros] elements
    threshold1 = x1[num_retain].item()

    x, _ = torch.sort(torch.abs(weight.data.view(-1)))      # Method used in GraNet implementation
    threshold = x[k-1].item()

    # assert threshold1 == threshold, f"Threshold mismatch {threshold1} != {threshold}"

    # Create a mask such that the indices of the top k elements are 1 and the rest are 0
    return (torch.abs(weight.data) > threshold).float()

def gradient_growth(mask, weight, num_pruned):
    grad = weight.grad
    masked = (mask == 0.0).float()
    grad = grad * masked.to(grad.device)
    x, _ = torch.sort(torch.abs(grad.view(-1)), descending=True)
    threshold = x[num_pruned].item()
    grad_mask = (torch.abs(grad) > threshold).float()
    grad_mask = grad_mask.to(mask.device)

    # If threshold is 0 (can occur in the ultra-sparse regime), logic to ensure that non-zero elements remain the same before and after prune and grow
    # Randomly select elements from the mask to regrow
    num_grown = grad_mask.sum().item()
    if num_pruned > num_grown:
        num_remain = int(num_pruned - num_grown)
        new_grad_mask = (mask == 0.0).float() * (grad_mask == 0.0).float()
        indices = torch.nonzero(new_grad_mask)
        shuffled_indices = indices[torch.randperm(indices.size(0))]
        regrow_indices = shuffled_indices[:num_remain]
        grad_mask[regrow_indices[:, 0], regrow_indices[:, 1], regrow_indices[:, 2], regrow_indices[:, 3]] = 1.0

    assert grad_mask.sum().item() == num_pruned, f"Number of elements in grad_mask {grad_mask.sum().item()} != {num_pruned}"
        
    return grad_mask

def pruning(model, args, prune_step):
    """
    Prunes the model based on the defined arguments and current training step. Retains the weights but updates the mask.

    Args:
        model (torch.nn.Module): The model to be pruned.
        args (argparse.Namespace): The arguments containing pruning parameters.
        step (int): The current training step.

    Returns:
        None]
    """

    final_step = int(args.dst_final_prune_epoch / args.update_frequency)
    init_step = int(args.dst_init_prune_epoch / args.update_frequency)
    total_step = final_step - init_step

    print('******************************************************')
    print(f'Pruning Progress is {prune_step - init_step} / {total_step}')
    print('******************************************************')

    # Conditions for Pruning. Omit if you want to prune based on epochs instead of steps
    assert final_step > init_step, 'Final step must be greater than initial step'
    assert args.er_sparse_init > args.final_density, 'Initial density must be greater than final density'

    if args.dst_prune_const:    # Whether to prune at a constant rate or anneal the pruning rate based on iteration
        # Use a constant Dynamic Prune Rate
        curr_prune_rate = args.dst_const_prune_rate
    else:
        # Using a version of eqn. 1 from section 4.1 the GraNet paper
        if prune_step >= init_step and prune_step <= final_step - 1:
            prune_decay = (1 - ((prune_step - init_step) / total_step)) ** 3
            curr_prune_rate = (1 - args.er_sparse_init) + (args.er_sparse_init - args.final_density) * (
                    1 - prune_decay)
        else:
            return
    
    weight_abs = []
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvER)):
            weight_abs.append(torch.abs(m.weight))

    all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
    num_params_to_keep = int(len(all_scores) * (1 - curr_prune_rate))

    x, _ = torch.topk(all_scores, num_params_to_keep)
    threshold = x[-1]

    total_size = 0
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvER)):
            total_size += torch.nonzero(m.weight.data).size(0)
            zero = torch.tensor([0.]).to(m.weight.device)
            one = torch.tensor([1.]).to(m.weight.device)
            m.mask = torch.where((torch.abs(m.weight) > threshold), one, zero)

    print('Total Model parameters:', total_size)

    sparse_size = 0
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvER)):
            sparse_size += torch.nonzero(m.mask).size(0)

    print('% Pruned: {0}'.format((total_size - sparse_size) / total_size))

def truncate_weights(model, prune_rate):
    """
    Creates an updated mask based on the weights of the model using the Prune and Grow algorithm.

    Args:
        model (nn.Module): The model to truncate the weights of.
        args: Additional arguments.
        step: The current step.
        prune_rate (float): The rate at which to prune the weights.

    Returns:
        None
    """

    num_nonzeros = []
    num_zeros = []

    # Model statistics before Prune and Grow
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvER)):
            num_nonzeros.append(m.mask.sum().item())
            num_zeros.append(m.mask.numel() - num_nonzeros[-1])

    # Prune
    layer = 0
    num_pruned = []
    updated_mask = []
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvER)):
            mask = m.mask
            new_mask = magnitude_death(mask, m.weight, prune_rate, num_nonzeros[layer], num_zeros[layer])
            new_mask = new_mask.to(m.mask.device)
            num_pruned.append(int(num_nonzeros[layer] - new_mask.sum().item()))
            updated_mask.append(new_mask)
            m.mask = new_mask
            layer += 1

    # Grow
    layer = 0
    total_retained = 0.0
    total_params = 0.0
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvER)):
            mask = m.mask
            assert torch.equal(mask, updated_mask[layer]), f"Mask mismatch {mask.sum().item()} != {updated_mask[layer].sum().item()}"

            new_mask = gradient_growth(mask, m.weight, num_pruned[layer])
            m.mask = new_mask.to(mask.device) + mask

            # Sanity Checks
            assert torch.all(m.mask <= 1.0), "Mask value greater than 1.0"
            # assert m.mask.sum().item() - updated_mask[layer].sum().item() == num_pruned[layer], f"Layer {layer}: Name:{n} -> Pruning and Regeneration mismatch. {m.mask.sum().item()} != {num_nonzeros[layer]}"

            print(f"{n}: Density: {m.mask.sum().item() / m.mask.numel()}")
            total_retained += m.mask.sum().item()
            total_params += m.mask.numel()
            layer += 1

    print(f"Overall Density: {total_retained / total_params}")

def log_mask_sparsity(model, writer, step, train_loader_len):
    # Log the mask sparsity
    for n, m in model.named_modules():
        if isinstance(m, STRConvER):
            maskSparsity, _ = m.getMaskSparsity()
            writer.add_scalar("mask_sparsity/{}".format(n), maskSparsity, (step // train_loader_len))


def step(model, args, step, train_loader_len, decay_scheduler=None):

    # Decay the pruning rate as the network trains. Used in truncate_weights()
    if args.dst_prune_const:
        prune_rate = args.dst_const_prune_rate
    elif decay_scheduler is not None:
        decay_scheduler.step()
        prune_rate = decay_scheduler.get_dr()

    
    if args.dst_method == 'GraNet':
        # Set up a warmup period since we want GraNet to come into the picture in the high sparsity regime
        if step >= (args.dst_init_prune_epoch * train_loader_len) and step % (args.update_frequency * train_loader_len) == 0:
            prune_step = step // (args.update_frequency * train_loader_len)
            pruning(model, args, prune_step)
            truncate_weights(model, prune_rate)

    elif args.dst_method == 'prune_and_grow':
        if step >= (args.dst_init_prune_epoch * train_loader_len) and step % (args.update_frequency * train_loader_len) == 0 and step <= (args.dst_final_prune_epoch * train_loader_len):
            # Omit GraNet pruning and only perform neuroregeneration
            truncate_weights(model, prune_rate)


def train(train_loader, model, criterion, optimizer, epoch, args, writer, prev_epochs=0):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{prev_epochs + epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True).long()

        # compute output
        output = model(images)

        loss = criterion(output, target.view(-1))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_steps = ((prev_epochs + epoch) * num_batches) + i
        if args.dst_method != 'None' and args.update_frequency is not None:
            step(model, args, num_steps, num_batches, None)

        log_mask_sparsity(model, writer, num_steps, num_batches)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * (prev_epochs + epoch) + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch, prev_epochs=0):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True).long()

            # compute output
            output = model(images)

            loss = criterion(output, target.view(-1))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=prev_epochs + epoch)

    return top1.avg, top5.avg

