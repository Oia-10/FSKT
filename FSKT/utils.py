import os
import random
import datetime
import numpy as np
import torch


def set_seed(seed=1010):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Logger:
    def __init__(self, result_dir, opt):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(result_dir, opt.dataset, timestamp)
        os.makedirs(self.save_dir, exist_ok=True)

        self.log_path = os.path.join(self.save_dir, 'log.txt')
        self.opt = opt
        self._write_header()

    def _write_header(self):
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("FSKT Training Log\n")
            f.write("="*60 + "\n")
            for key, val in vars(self.opt).items():
                if key != 'paths':
                    f.write(f"{key}: {val}\n")
            f.write("="*60 + "\n\n")

    def log(self, msg):
        print(msg)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

    def log_epoch(self, fold, epoch, train_metrics, valid_metrics, best_auc):
        msg = (f"Fold {fold} | Epoch {epoch:3d} | "
               f"Train AUC: {train_metrics['auc']:.4f} Loss: {train_metrics['loss']:.4f} | "
               f"Valid AUC: {valid_metrics['auc']:.4f} | "
               f"Best: {best_auc:.4f}")
        self.log(msg)

    def log_test(self, fold, metrics):
        metrics_str = ' | '.join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items()])
        self.log(f"Fold {fold} Test | {metrics_str}")

    def log_final(self, all_results):
        self.log("\n" + "="*60)
        self.log("Final Results (5-fold CV)")
        self.log("="*60)

        summary = {}
        for metric in ['auc', 'acc', 'rmse', 'r2']:
            vals = [r[metric] for r in all_results]
            summary[metric] = (np.mean(vals), np.std(vals))

        result_str = ' | '.join([
            f"{k.upper()}: {v[0]:.4f}±{v[1]:.4f}"
            for k, v in summary.items()
        ])
        self.log(result_str)

        return summary

    def save_model(self, model, fold):
        path = os.path.join(self.save_dir, f'model_fold{fold}.pt')
        torch.save(model.state_dict(), path)
        return path
