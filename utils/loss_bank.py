import torch
import torch.nn.functional as F

def message_loss_fn(recover_message, message, device):
    loss_fn = torch.nn.MSELoss(reduce=True)
    loss = loss_fn(recover_message, message)
    return loss.to(device)

def stego_loss_fn(stego, cover, device):
    loss_fn = torch.nn.MSELoss(reduce=True)
    loss = loss_fn(stego, cover)
    return loss.to(device)

def do_loss(recover_message, message, deflation_weight, inflation_weight, safe_value, device):
    sign = torch.sign(message)
    gap = torch.zeros_like(message).to(device) - recover_message

    results = gap * sign

    fault_mask = torch.where(results > 0, torch.ones_like(results), torch.zeros_like(results))

    true_mask = 1 - fault_mask
    safe_mask = torch.where(results > safe_value, torch.ones_like(results), torch.zeros_like(results))

    loss = deflation_weight * torch.sum(results * fault_mask) + inflation_weight * torch.sum(results * (true_mask * safe_mask))
    return loss.to(device)