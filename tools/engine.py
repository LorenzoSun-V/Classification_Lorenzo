import time
import torch
from logzero import logger as log
from utils.model_utils import AverageMeter, get_fscore, accuracy


def train_one_epoch(loader, model, optimizer, device, print_freq, epoch, epochs, one_hot, local_rank):
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

        # print(logits)
        # print(loss)

        optimizer.zero_grad()
        if one_hot:
            loss.mean().backward()
        else:
            loss.backward()
        optimizer.step()

        if one_hot:
            prec1, recall, fscore = get_fscore(y_true=targets.cpu(), y_pred=logits.sigmoid().cpu() > 0.5)
        else:
            prec1 = accuracy(logits.data, targets, topk=(1,))[0]
            prec1 = prec1.item()

        losses.update(loss.sum().data.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx % print_freq == 0) and (local_rank in [-1, 0]):
            log.info(f"TRAIN | Epoch: [{epoch}/{epochs}][{batch_idx}/{len(loader)}] | Time: {data_time.val:.3f}/{batch_time.val:.3f} | LOSS: {losses.val:.3f} | TOP1: {top1.val:.3f} ")

    if local_rank in [-1, 0]:
        log.info(f"TRAIN | Epoch: [{epoch}/{epochs}] | Time: {batch_time.sum:.3f} | LOSS: {losses.avg:.3f} | TOP1: {top1.avg:.3f} \n")

    return losses.avg, top1.avg


@torch.no_grad()
def val_one_epoch(loader, model, device, one_hot, local_rank):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    for batch_id, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        logits, loss = model(inputs, targets)
        if one_hot:
            prec1, recall, fscore = get_fscore(y_true=targets.cpu(), y_pred=logits.sigmoid().cpu() > 0.5)
        else:
            prec1 = accuracy(logits.data, targets, topk=(1,))[0]
            prec1 = prec1.item()

        loss = loss.sum().item()
        losses.update(loss, inputs.size(0))
        top1.update(prec1, inputs.size(0))

    if local_rank in [-1, 0]:
        log.info(f"val loss: {losses.avg:.4f} | TOP1: {top1.avg:.4f} \n")
    return losses.avg, top1.avg
