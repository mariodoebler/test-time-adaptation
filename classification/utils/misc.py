import torch
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def ema_update_model(model_to_update, model_to_merge, momentum, device, update_all=False):
    if momentum < 1.0:
        for param_to_update, param_to_merge in zip(model_to_update.parameters(), model_to_merge.parameters()):
            if param_to_update.requires_grad or update_all:
                param_to_update.data = momentum * param_to_update.data + (1 - momentum) * param_to_merge.data.to(device)
    return model_to_update


def print_memory_info():
    logger.info('-' * 40)
    mem_dict = {}
    for metric in ['memory_allocated', 'max_memory_allocated', 'memory_reserved', 'max_memory_reserved']:
        mem_dict[metric] = eval(f'torch.cuda.{metric}()')
        logger.info(f"{metric:>20s}: {mem_dict[metric] / 1e6:10.2f}MB")
    logger.info('-' * 40)
    return mem_dict
