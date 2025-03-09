import time
import torch
import tqdm
import math
import torchvision

from utils.eval_utils import accuracy
from utils.conv_type import STRConv, STRConvER
from utils.logging import AverageMeter, ProgressMeter
from args import args


__all__ = ["train", "validate"]

# def magnitude_death(mask, weight, prune_rate, num_nonzeros, num_zeros):
#     num_remove = math.ceil(prune_rate * num_nonzeros)
#     # num_retain = math.ceil(num_nonzeros - num_remove)  # [(1 - prune_rate) * num_nonzeros]

#     if num_remove == 0.0:
#         return weight.data != 0.0
    
#     k = math.ceil(num_zeros + num_remove)

#     # x1, _ = torch.sort(torch.abs(weight.data[mask == 1.0]), descending=True) # Alternate logic to extract the top [(1 - prune_rate) * num_nonzeros] elements
#     # threshold1 = x1[num_retain].item()

#     x, _ = torch.sort(torch.abs(weight.data.view(-1)))      # Method used in GraNet implementation
#     threshold = x[k-1].item()

#     # assert threshold1 == threshold, f"Threshold mismatch {threshold1} != {threshold}"

#     # Create a mask such that the indices of the top k elements are 1 and the rest are 0
#     return (torch.abs(weight.data) > threshold).float()

# def gradient_growth(mask, weight, num_pruned):
#     grad = weight.grad
#     masked = (mask == 0.0).float()
#     grad = grad * masked.to(grad.device)
#     x, _ = torch.sort(torch.abs(grad.view(-1)), descending=True)
#     threshold = x[num_pruned].item()
#     grad_mask = (torch.abs(grad) > threshold).float()
#     grad_mask = grad_mask.to(mask.device)

#     # If threshold is 0 (can occur in the ultra-sparse regime), logic to ensure that non-zero elements remain the same before and after prune and grow
#     # Randomly select elements from the mask to regrow
#     num_grown = grad_mask.sum().item()
#     if num_pruned > num_grown:
#         num_remain = int(num_pruned - num_grown)
#         new_grad_mask = (mask == 0.0).float() * (grad_mask == 0.0).float()
#         indices = torch.nonzero(new_grad_mask)
#         shuffled_indices = indices[torch.randperm(indices.size(0))]
#         regrow_indices = shuffled_indices[:num_remain]
#         grad_mask[regrow_indices[:, 0], regrow_indices[:, 1], regrow_indices[:, 2], regrow_indices[:, 3]] = 1.0

#     assert grad_mask.sum().item() == num_pruned, f"Number of elements in grad_mask {grad_mask.sum().item()} != {num_pruned}"
        
#     return grad_mask

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
            m.er_mask = torch.where((torch.abs(m.weight) > threshold), one, zero)

    print('Total Model parameters:', total_size)

    sparse_size = 0
    for n, m in model.named_modules():
        if isinstance(m, (STRConv, STRConvER)):
            sparse_size += torch.nonzero(m.er_mask).size(0)

    print('% Pruned: {0}'.format((total_size - sparse_size) / total_size))

# def truncate_weights(model, prune_rate):
#     """
#     Creates an updated mask based on the weights of the model using the Prune and Grow algorithm.

#     Args:
#         model (nn.Module): The model to truncate the weights of.
#         args: Additional arguments.
#         step: The current step.
#         prune_rate (float): The rate at which to prune the weights.

#     Returns:
#         None
#     """

#     num_nonzeros = []
#     num_zeros = []

#     # Model statistics before Prune and Grow
#     for n, m in model.named_modules():
#         if isinstance(m, (STRConv, STRConvER)):
#             num_nonzeros.append(m.mask.sum().item())
#             num_zeros.append(m.mask.numel() - num_nonzeros[-1])

#     # Prune
#     layer = 0
#     num_pruned = []
#     updated_mask = []
#     for n, m in model.named_modules():
#         if isinstance(m, (STRConv, STRConvER)):
#             mask = m.mask
#             new_mask = magnitude_death(mask, m.weight, prune_rate, num_nonzeros[layer], num_zeros[layer])
#             new_mask = new_mask.to(m.mask.device)
#             num_pruned.append(int(num_nonzeros[layer] - new_mask.sum().item()))
#             updated_mask.append(new_mask)
#             m.mask = new_mask
#             layer += 1

#     # Grow
#     layer = 0
#     total_retained = 0.0
#     total_params = 0.0
#     for n, m in model.named_modules():
#         if isinstance(m, (STRConv, STRConvER)):
#             mask = m.mask
#             assert torch.equal(mask, updated_mask[layer]), f"Mask mismatch {mask.sum().item()} != {updated_mask[layer].sum().item()}"

#             new_mask = gradient_growth(mask, m.weight, num_pruned[layer])
#             m.mask = new_mask.to(mask.device) + mask

#             # Sanity Checks
#             assert torch.all(m.mask <= 1.0), "Mask value greater than 1.0"
#             # assert m.mask.sum().item() - updated_mask[layer].sum().item() == num_pruned[layer], f"Layer {layer}: Name:{n} -> Pruning and Regeneration mismatch. {m.mask.sum().item()} != {num_nonzeros[layer]}"

#             print(f"{n}: Density: {m.mask.sum().item() / m.mask.numel()}")
#             total_retained += m.mask.sum().item()
#             total_params += m.mask.numel()
#             layer += 1

#     print(f"Overall Density: {total_retained / total_params}")

# def magnitude_death(mask, sparse_weights, active_mask, prune_rate, num_nonzeros):
#     """Prune weights that survive STR but have smallest magnitudes"""
#     num_remove = math.ceil(prune_rate * num_nonzeros)
#     if num_remove == 0 or num_nonzeros == 0:
#         return mask.clone()
    
#     # Get magnitudes of active weights (non-zero after STR)
#     active_magnitudes = torch.abs(sparse_weights[active_mask])
#     sorted_mags, _ = torch.sort(active_magnitudes)
#     threshold = sorted_mags[num_remove-1] if num_remove <= len(sorted_mags) else 0
    
#     # Create prune mask for active weights below threshold
#     prune_mask = (torch.abs(sparse_weights) <= threshold) & active_mask
#     new_mask = mask.clone()
#     new_mask[prune_mask] = 0
#     return new_mask

# def gradient_growth(mask, weight, num_pruned):
#     """Regrow weights with largest gradients while allowing gradient flow"""
#     grad = weight.grad
#     current_mask = mask.bool()
    
#     # Consider only masked weights for regrowth
#     candidate_grads = grad[~current_mask]
#     if candidate_grads.numel() == 0:
#         return mask.clone(), None
    
#     # Find threshold for top gradients
#     sorted_grads = torch.sort(torch.abs(candidate_grads), descending=True)[0]
#     threshold_idx = min(num_pruned, len(sorted_grads)) - 1                                                                                                                                                                                                                                                                                                                                                                                                                             
#     threshold = sorted_grads[threshold_idx] if threshold_idx >= 0 else 0
    
#     # Create regrowth mask
#     grad_mask = (torch.abs(grad) >= threshold) & ~current_mask
#     new_mask = mask.clone()
#     new_mask[grad_mask] = 1
    
#     return new_mask, grad_mask

def get_regrowth_candidates(layer):
    """Get valid regrowth candidates: non-ER, non-grown, STR-zeroed"""
    device = layer.er_mask.device
    str_zero = (layer.sparseFunction(layer.weight, layer.sparseThreshold) == 0).to(device)
    return (layer.er_mask.bool().to(device) & 
           ~layer.dynamic_growth_mask.to(device) & 
           str_zero)

def reinitialize_weights(layer, mask):
    """Reinitialize weights with gradient-aligned signs and average magnitude"""
    device = layer.weight.device
    # Calculate average magnitude from current active weights
    mask = mask.to(device)
    er_mask_gpu = layer.er_mask.bool().to(device)
    dynamic_prune_mask_gpu = layer.dynamic_prune_mask.to(device)
    dynamic_growth_mask_gpu = layer.dynamic_growth_mask.to(device)
    
    active_weights = layer.sparseFunction(layer.weight, layer.sparseThreshold)
    active_weights = active_weights[(er_mask_gpu & ~dynamic_prune_mask_gpu) |
        dynamic_growth_mask_gpu]
    active_weight_magnitude = layer.weight[(er_mask_gpu & ~dynamic_prune_mask_gpu) |
        dynamic_growth_mask_gpu]

    active_non_zero = active_weight_magnitude[active_weights != 0]
    avg_mag = active_non_zero.abs().mean() if len(active_non_zero) > 0 else (torch.sigmoid(layer.sparseThreshold) + 0.01).to(device)
    
    # Determine signs from gradients
    grad_signs = -torch.sign(layer.weight.grad[mask]).to(device)
    
    # Handle zero gradients
    zero_grad = (grad_signs == 0).to(device)
    grad_signs[zero_grad] = torch.where(
        # Ensure rand is on the same device
        torch.rand(zero_grad.sum().item(), device=device) > 0.5,
        torch.tensor(1.0, device=device),
        torch.tensor(-1.0, device=device)
    )

    # Apply reinitialization
    with torch.no_grad():
        layer.weight.data[mask] = (grad_signs * avg_mag)

def update_pruned_weights(layer, mask):
    device = layer.weight.device
    mask = mask.to(device)
    # Update pruned weight magnitudes so that they become candidates to regrow
    mag = (torch.sigmoid(layer.sparseThreshold) - 1e-6).to(device)
    mag_signs = torch.sign(layer.weight.data[mask]).to(device)

    with torch.no_grad():
        layer.weight.data[mask] = (mag_signs * mag)

def magnitude_death(layer, prune_rate):
    """Prune weights from ER mask that survived STR, limited by regrowth candidates"""
    # Get candidate weights for pruning
    device = layer.er_mask.device
    active_mask = (layer.er_mask.bool() & ~layer.dynamic_prune_mask).to(device)
    str_active = (layer.sparseFunction(layer.weight, layer.sparseThreshold) != 0).to(device)
    candidates = (active_mask & str_active).to(device)
    num_active = candidates.sum().item()
    
    # Calculate available regrowth candidates
    regrowth_candidates = get_regrowth_candidates(layer).to(device)
    num_available_regrowth = regrowth_candidates.sum().item()
    
    # Determine actual prune count
    intended_prune = math.ceil(prune_rate * num_active)
    actual_prune = min(intended_prune, num_available_regrowth)
    
    if actual_prune == 0 or num_active == 0:
        return 0, None

    # Find smallest magnitude candidates
    magnitudes = torch.abs(layer.weight.data[candidates]).to(device)
    threshold = torch.kthvalue(magnitudes.flatten(), actual_prune)[0]
    
    # Update prune mask
    prune_mask = ((torch.abs(layer.weight.data) <= threshold).to(device) & candidates).to(device)
    layer.dynamic_prune_mask |= prune_mask
    
    return actual_prune, prune_mask

def gradient_growth(layer, num_regrow):
    """Regrow weights outside ER mask zeroed by STR, with gradient priority"""
    if layer.weight.grad is None:
        return
    device = layer.er_mask.device
    candidates = get_regrowth_candidates(layer).to(device)
    num_candidates = candidates.sum().item()
    
    # Gradient-based selection
    grad_mags = torch.abs(layer.weight.grad[candidates])
    num_grad = min(num_regrow, num_candidates)
    
    if num_grad > 0:
        threshold = torch.kthvalue(grad_mags.flatten(), num_candidates - num_grad + 1)[0]
        grad_mask = (torch.abs(layer.weight.grad) > threshold).to(device) & candidates            
    else:
        grad_mask = torch.zeros_like(candidates, device=device)

    # Random selection for remainder
    remaining = num_regrow - grad_mask.sum()
    if remaining > 0:
        available = candidates & ~grad_mask
        indices = torch.nonzero(available.flatten(), as_tuple=False).to(device)
        rand_idx = indices[torch.randperm(len(indices))[:remaining]]
        random_mask = torch.zeros_like(candidates.flatten(), device=device)
        random_mask[rand_idx] = True
        random_mask = random_mask.reshape(candidates.shape)
    else:
        random_mask = torch.zeros_like(candidates, device=device)

    # Update growth mask and reinitialize
    layer.dynamic_growth_mask |= (grad_mask | random_mask)
    if args.is_reinitialze_grown_weights:
        reinitialize_weights(layer, grad_mask | random_mask)

def resolve_mask_conflicts(layer):
    """Ensure no weight exists in both prune and growth masks"""
    conflict_mask = layer.dynamic_prune_mask & layer.dynamic_growth_mask
    layer.dynamic_prune_mask[conflict_mask] = False
    layer.dynamic_growth_mask[conflict_mask] = False
    return conflict_mask.sum().item()

def create_mask_visualization(layer):
    """Create comprehensive visualization for a single layer"""
    device = layer.er_mask.device
    with torch.no_grad():
        # Get masks and weights
        sparse_weights = layer.sparseFunction(layer.weight, layer.sparseThreshold).to(device)
        er_mask = layer.er_mask.bool().to(device)
        str_active = (sparse_weights != 0).to(device)
        
        # Initialize RGB visualization tensor [3, C, H, W]
        vis = torch.zeros(3, *layer.weight.shape[2:], dtype=torch.uint8, device=device)
        vis_prune_regrow_mask = torch.zeros(3, *layer.weight.shape[2:], dtype=torch.uint8, device=device)
        
        # Color mapping with priority
        colors = {
            'er_masked': (0,255,0),     # Green - never affected
            'growth_candidates': (255, 255, 0),     # Yellow
            'prune_candidates': (0, 0, 255),    # Blue
            'regrown': (255, 0, 0),  # Red
            'pruned': (120, 81, 169) # Purple
        }

        
        regrowth_candidates = get_regrowth_candidates(layer).to(device)

        active_mask = (layer.er_mask.bool() & ~layer.dynamic_prune_mask).to(device)
        str_active = (sparse_weights != 0).to(device)
        prune_candidates = (active_mask & str_active).to(device)

        candidate_mask = [
            (regrowth_candidates[0,0], colors['growth_candidates']),
            (prune_candidates[0,0], colors['prune_candidates']),
            (regrowth_candidates[0,0] & prune_candidates[0,0], colors['regrown'])
        ]

        # Apply colors with priority
        for mask, color in candidate_mask:
            color_tensor = torch.tensor(color, dtype=vis_prune_regrow_mask.dtype, device=device)
            vis_prune_regrow_mask[:, mask] = color_tensor[:, None]

        # Create masks with priority order
        masks = [
            (~er_mask[0,0], colors['er_masked']),
            (regrowth_candidates[0,0], colors['growth_candidates']),
            (prune_candidates[0,0], colors['prune_candidates']),
            (layer.dynamic_growth_mask[0,0].to(device), colors['regrown']),
            (layer.dynamic_prune_mask[0,0].to(device), colors['pruned'])
        ]

        # Apply colors with priority
        for mask, color in masks:
            color_tensor = torch.tensor(color, dtype=vis.dtype, device=device)
            vis[:, mask] = color_tensor[:, None]

    return vis, sparse_weights, vis_prune_regrow_mask

def log_layer_masks(writer, model, global_step):
    """Log all layer masks with multiple views"""
    for name, module in model.named_modules():
        if isinstance(module, STRConvER):
            device = module.er_mask.device
            # Individual filter view
            vis, sparse_weights, vis_prune_regrow = create_mask_visualization(module)
            writer.add_image(f"Masks/{name}/Filter0", vis, global_step)
            writer.add_image(f"Masks/{name}/Filter0_prune_regrow_candidates", vis_prune_regrow, global_step)
            
            # Full layer grid view
            grid = torch.stack([create_mask_visualization(module)[0] 
                              for _ in range(min(16, module.weight.size(0)))], dim=0).to(device)
            grid = torchvision.utils.make_grid(grid.float()/255, nrow=4)
            writer.add_image(f"Masks/{name}/Grid", grid, global_step)
            
            # Density metrics
            writer.add_scalar(f"Density/{name}/ER_Active", 
                             (module.er_mask.bool() & ((sparse_weights != 0).to(device))).float().mean(), global_step)
            writer.add_scalar(f"Density/{name}/Regrown", 
                             module.dynamic_growth_mask.float().mean(), global_step)
            writer.add_scalar(f"Density/{name}/Pruned",
                             module.dynamic_prune_mask.float().mean(), global_step)
            
def log_gradient_distributions(writer, model, global_step):
    for name, module in model.named_modules():
        if isinstance(module, STRConvER):
            if module.weight.grad is None:
                continue
            device = module.er_mask.device
            masks = {
                'ER_Active': (module.er_mask.bool() & (module.sparseFunction(module.weight, module.sparseThreshold).to(device) != 0)).to(device),
                'Pruned': module.dynamic_prune_mask.to(device),
                'Regrown': module.dynamic_growth_mask.to(device)
            }
            
            for mask_type, mask in masks.items():
                if mask is None:
                    continue
                grads = module.weight.grad[mask].to(device)
                if grads.numel() > 0:
                    writer.add_histogram(f"Gradients/{name}/{mask_type}", grads, global_step)


def truncate_weights(model, prune_rate, writer, global_step=1):
    """Main neuroregeneration entry point"""
    for module in model.modules():
        if not isinstance(module, STRConvER):
            continue

        resolve_mask_conflicts(module)
            
        # 1. Calculate actual prune count based on regrowth capacity
        num_pruned, prune_mask = magnitude_death(module, prune_rate)
        
        # 2. Perform gradient-based regrowth
        if num_pruned > 0:
            gradient_growth(module, num_pruned)
            if args.is_update_pruned_weights:
                update_pruned_weights(module, prune_mask)


        # Create and log visualization
        log_layer_masks(writer, model, global_step)
        log_gradient_distributions(writer, model, global_step)

# def perform_neuroregeneration(model, prune_rate, epsilon=1e-8):
#     """
#     Perform Neuroregeneration on the model. This function is used in the prune_and_grow method.

#     Args:
#         model (nn.Module): The model to perform neuroregeneration on.
#         prune_rate (float): The rate at which to prune the weights.
#         epsilon (float): A small value to prevent division by zero.

#     Returns:
#         None
#     """
#     total_active = 0
#     total_params = 0
    
#     for name, module in model.named_modules():
#         if not isinstance(module, (STRConv, STRConvER)):
#             continue
            
#         # Compute STR-transformed weights
#         sparse_weights = module.sparseFunction(
#             module.weight,
#             module.sparseThreshold,
#             module.activation,
#             module.f
#         )
        
#         # Identify actually active weights (masked and surviving STR)
#         active_mask = (module.mask == 1) & (sparse_weights != 0)
#         num_active = active_mask.sum().item()
        
#         # Prune lowest magnitude active weights
#         new_mask = magnitude_death(
#             module.mask, sparse_weights, active_mask,
#             prune_rate, num_active
#         )
#         num_pruned = int((module.mask.sum() - new_mask.sum()).item())
        
#         # Regrow highest gradient weights
#         new_mask, grad_mask = gradient_growth(new_mask, module.weight, num_pruned)
        
#         # Reinitialize regenerated weights
#         if grad_mask is not None and grad_mask.any():
#             # Calculate average magnitude of current active weights
#             current_active = sparse_weights * new_mask
#             active_non_zero = current_active[current_active != 0]
#             avg_mag = active_non_zero.abs().mean() if len(active_non_zero) > 0 else 0.01
            
#             # Initialize with gradient-aligned signs
#             new_weights = module.weight.data[grad_mask]
#             grad_signs = -torch.sign(module.weight.grad[grad_mask])
            
#             # Handle zero gradients randomly
#             zero_grads = (grad_signs == 0)
#             grad_signs[zero_grads] = torch.where(
#                 torch.rand(zero_grads.sum().item()) > 0.5,
#                 torch.ones_like(grad_signs[zero_grads]),
#                 -torch.ones_like(grad_signs[zero_grads])
#             )
            
#             with torch.no_grad():
#                 module.weight.data[grad_mask] = grad_signs * avg_mag
        
#         # Update module mask
#         module.mask = new_mask
        
#         # Track sparsity
#         layer_active = new_mask.sum().item()
#         layer_params = new_mask.numel()
#         total_active += layer_active
#         total_params += layer_params
#         print(f"{name}: Density {layer_active/layer_params:.3f}")

#     print(f"Global density: {total_active/total_params:.4f}")
    
    

def log_mask_sparsity(model, writer, step, train_loader_len):
    # Log the mask sparsity
    for n, m in model.named_modules():
        if isinstance(m, STRConvER):
            maskSparsity, _ = m.getMaskSparsity()
            writer.add_scalar("mask_sparsity/{}".format(n), maskSparsity, (step // train_loader_len))


def step(model, args, step, train_loader_len, decay_scheduler=None, writer=None):

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
            truncate_weights(model, prune_rate,writer=writer, global_step=step)

    elif args.dst_method == 'prune_and_grow':
        if step >= (args.dst_init_prune_epoch * train_loader_len) and step % (args.update_frequency * train_loader_len) == 0 and step <= (args.dst_final_prune_epoch * train_loader_len):
            # Omit GraNet pruning and only perform neuroregeneration
            # truncate_weights(model, prune_rate)
            truncate_weights(model, prune_rate,writer=writer, global_step=step)


def train(train_loader, model, criterion, optimizer, epoch, args, writer, prev_epochs=0, device='cpu'):
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
            step(model, args, num_steps, num_batches, None, writer=writer)

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

