import os
import copy
import torch
from torch.optim.lr_scheduler import StepLR
from config import set_opt, RESULT_DIR
from utils import set_seed, EarlyStopping, Logger
from dataload import load_dataset, load_split, create_dataloader, load_q_matrix
from model import FSKT
from trainer import train, evaluate


def run_fold(fold, model, train_loader, valid_loader, test_loader,
             optimizer, scheduler, device, opt, logger):
    best_auc = 0.0
    best_model_state = None 
    early_stopping = EarlyStopping(patience=opt.early_stop_patience)

    for epoch in range(1, opt.epochs + 1):
        train_metrics = train(model, train_loader, optimizer, device, opt, scheduler)
        valid_metrics = evaluate(model, valid_loader, device, opt)

        if valid_metrics['auc'] > best_auc + 1e-3:
            best_auc = valid_metrics['auc']
            logger.save_model(model, fold)
            best_model_state = copy.deepcopy(model.state_dict())

        logger.log_epoch(fold, epoch, train_metrics, valid_metrics, best_auc)

        early_stopping(valid_metrics['auc'])
        if early_stopping.early_stop:
            logger.log(f"Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, device, opt)
    logger.log_test(fold, test_metrics)
    return test_metrics


def main():
    opt = set_opt()
    set_seed(opt.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = Logger(RESULT_DIR, opt)
    logger.log(f"Device: {device}")

    data = load_dataset(opt)
    Q_matrix = load_q_matrix(opt)
    test_uids = load_split(opt.paths['test_split'])
    test_loader = create_dataloader(data[data['uid'].isin(test_uids)], opt)

    all_results = []
    for fold in range(5):
        logger.log(f"\n{'='*50}\nFold {fold}\n{'='*50}")

        train_uids, valid_uids = load_split(opt.paths['train_valid_split'], cv=fold)
        assert set(train_uids).isdisjoint(valid_uids), f"Fold {fold}: train/valid uid overlap"
        assert set(train_uids).isdisjoint(test_uids), f"Fold {fold}: train/test uid overlap"
        assert set(valid_uids).isdisjoint(test_uids), f"Fold {fold}: valid/test uid overlap"

        train_loader = create_dataloader(data[data['uid'].isin(train_uids)], opt, shuffle=True)
        valid_loader = create_dataloader(data[data['uid'].isin(valid_uids)], opt)

        model = FSKT(
            num_q=opt.num_q,
            num_c=opt.num_c,
            embed_dim=opt.embed_dim,
            hidden_dim=opt.hidden_dim,
            Q_matrix=torch.from_numpy(Q_matrix).float(),
            dropout=opt.dropout,
            n_blocks=opt.n_blocks,
            n_heads=opt.n_heads,
            n_levels=opt.n_levels,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=opt.lr_decay_step, gamma=opt.lr_decay_rate)

        result = run_fold(fold, model, train_loader, valid_loader, test_loader,
                          optimizer, scheduler, device, opt, logger)
        all_results.append(result)

    summary = logger.log_final(all_results)
    logger.log(f"\nSave the results to: {logger.save_dir}")


if __name__ == '__main__':
    main()
