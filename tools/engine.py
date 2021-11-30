import time
import torch
from logzero import logger as log
from utils.model_utils import AverageMeter, get_fscore, accuracy
import torch.distributed as dist
import numpy as np


def reduce_tensor(inp):
    """
    Reduce the results from all processes so that
    process with rank 0 has the averaged results
    """
    world_size = float(dist.get_world_size())
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduce_inp = inp
        dist.reduce(reduce_inp, dst=0)
    return reduce_inp / world_size


def train_one_epoch(loader, USE_DDP, model, optimizer, device, print_freq, epoch, epochs, one_hot, local_rank):
    model.train()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(loader):
        # print(f"{local_rank}: {inputs.size}, {targets}")
        data_time.update(time.time() - end)
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        logits, loss = model(inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if one_hot:
            prec1, recall, fscore = get_fscore(y_true=targets.cpu(), y_pred=logits.sigmoid().cpu() > 0.5)
        else:
            prec1 = accuracy(logits.data, targets, topk=(1,))[0]
            prec1 = prec1.item()

        if USE_DDP:
            reduced_loss = reduce_tensor(loss)
            reduced_acc = reduce_tensor(torch.from_numpy(np.array(prec1)).cuda())
            top1.update(reduced_acc.data.item(), inputs.size(0))
        else:
            reduced_loss = loss
            reduced_acc = prec1
            top1.update(reduced_acc, inputs.size(0))
        losses.update(reduced_loss.data.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx % print_freq == 0) and (local_rank in [-1, 0]):
            log.info(f"TRAIN | Epoch: [{epoch}/{epochs}][{batch_idx}/{len(loader)}] | Time: {data_time.val:.3f}/{batch_time.val:.3f} | LOSS: {losses.val:.5f} | TOP1: {top1.val:.5f} ")

    if local_rank in [-1, 0]:
        log.info(f"TRAIN | Epoch: [{epoch}/{epochs}] | Time: {batch_time.sum:.3f} | LOSS: {losses.avg:.5f} | TOP1: {top1.avg:.5f} \n")

    return losses.avg, top1.avg


@torch.no_grad()
def val_one_epoch(loader, USE_DDP, model, device, one_hot, local_rank):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device=device, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)

            logits, loss = model(inputs, targets)
            if one_hot:
                prec1, recall, fscore = get_fscore(y_true=targets.cpu(), y_pred=logits.sigmoid().cpu() > 0.5)
            else:
                prec1 = accuracy(logits.data, targets, topk=(1,))[0]
                prec1 = prec1.item()

            if USE_DDP:
                reduced_loss = reduce_tensor(loss)
                reduced_acc = reduce_tensor(torch.from_numpy(np.array(prec1)).cuda())
                top1.update(reduced_acc.data.item(), inputs.size(0))
            else:
                reduced_loss = loss
                reduced_acc = prec1
                top1.update(reduced_acc, inputs.size(0))
            losses.update(reduced_loss.data.item(), inputs.size(0))

    if local_rank in [-1, 0]:
        log.info(f"val loss: {losses.avg:.5f} | TOP1: {top1.avg:.5f} \n")
    return losses.avg, top1.avg
