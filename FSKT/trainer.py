import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score


def train(model, dataloader, optimizer, device, opt, scheduler=None):
    model.train()
    y_trues, y_scores = [], []
    loss_mean = []
    criterion = nn.BCELoss(reduction='mean')

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        q = batch[1].to(device).long()      
        r = batch[2].to(device).long()       
        sm = batch[5].to(device).long()     

        optimizer.zero_grad()

        pred, extra = model(q, r, mask=sm.float())

        rshft = r[:, 1:]
        sm_shft = sm[:, 1:].bool()

        y1_masked = torch.masked_select(extra['pred_ac'], sm_shft)
        y2_masked = torch.masked_select(extra['pred_ae'], sm_shft)

        y_masked = torch.masked_select(pred, sm_shft)
        t_masked = torch.masked_select(rshft, sm_shft)

        bce_loss = criterion(y_masked, t_masked.float())
        bce_loss1 = criterion(y1_masked, t_masked.float())
        bce_loss2 = criterion(y2_masked, t_masked.float())

        reg_loss = extra.get('reg_loss', torch.tensor(0.0, device=pred.device))
        scd_loss = extra.get('scd_loss', {}).get('suff_total', torch.tensor(0.0, device=pred.device))

        loss = bce_loss1 + bce_loss2 + reg_loss + opt.lambda1 * scd_loss
        loss_mean.append(bce_loss.item())

        loss.backward()

        if opt.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

        optimizer.step()

        y_scores.extend(y_masked.detach().cpu().numpy().tolist())
        y_trues.extend(t_masked.detach().cpu().numpy().tolist())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    if scheduler is not None:
        scheduler.step()

    return _compute_metrics(y_trues, y_scores, loss_mean)


def evaluate(model, dataloader, device, opt):
    model.eval()
    y_trues, y_scores = [], []
    loss_mean = []
    criterion = nn.BCELoss(reduction='mean')

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            q = batch[1].to(device).long()
            r = batch[2].to(device).long()
            sm = batch[5].to(device).long()

            pred, extra = model(q, r, mask=sm.float())

            rshft = r[:, 1:]
            sm_shft = sm[:, 1:].bool()

            y_masked = torch.masked_select(pred, sm_shft)
            t_masked = torch.masked_select(rshft, sm_shft)

            loss = criterion(y_masked, t_masked.float())
            loss_mean.append(loss.item())

            y_scores.extend(y_masked.cpu().numpy().tolist())
            y_trues.extend(t_masked.cpu().numpy().tolist())

    metrics = _compute_metrics(y_trues, y_scores, loss_mean)

    return metrics


def _compute_metrics(ts, ps, loss_mean):
    try:
        auc = roc_auc_score(ts, ps)
    except ValueError:
        auc = 0.5  

    prelabels = [1 if p >= 0.5 else 0 for p in ps]
    acc = accuracy_score(ts, prelabels)
    mse = mean_squared_error(ts, ps)
    rmse = np.sqrt(mse)
    r2 = r2_score(ts, ps) if len(set(ts)) > 1 else 0.0

    return {
        'auc': auc,
        'acc': acc,
        'loss': np.mean(loss_mean),
        'rmse': rmse,
        'r2': r2
    }
