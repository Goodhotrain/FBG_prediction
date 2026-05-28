import os
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
from opt import parse_opts
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import random
import numpy as np 
random.seed(42)
np.random.seed(42)
import math
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import logging
from collections import Counter
from common.optimizer import get_optimizer, get_scheduler
from common.config import record_config
from data.dataset import FBGDataset
from models.model import FBGPredictor
import pandas as pd

def compute_weighted_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    class_counts = Counter(y_true)
    total = len(y_true)

    weighted_acc = 0.0
    for cls in class_counts:
        cls_mask = (y_true == cls)
        correct = (y_pred[cls_mask] == y_true[cls_mask]).sum()
        acc_c = correct / cls_mask.sum()  # per-class accuracy
        weight = cls_mask.sum() / total   # per-class weight
        weighted_acc += acc_c * weight

    return weighted_acc

def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    return total_params

class Run(object):
    def __init__(self):
        self.args = parse_opts()
        current_seed = torch.initial_seed()
        self.current_step = 0
        self.current_test_step = 0
        self.last_map = 0
        self.best_acc = 0
        if torch.cuda.is_available():
            self.args.device = torch.device('cuda:0')
        else:
            self.args.device = torch.device('cpu')
        now = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
        self.args.result_path = os.path.join(self.args.result_path, now)
        os.makedirs(self.args.result_path, exist_ok=True)
        record_config(self.args)
        logging.basicConfig(level=logging.INFO, filename=f'{self.args.result_path}/experiment.log', filemode='a')
        logging.info(f"Random seed set to: {current_seed}")
        self.model = FBGPredictor().to(self.args.device)
        get_model_parameters(self.model)
        self.optimizer = get_optimizer(self.args, self.model)
        self.scheduler = get_scheduler(self.args, self.optimizer)
        self.train_data = FBGDataset(self.args, subset='train')
        self.valid_data = FBGDataset(self.args, subset='test')
        self.train_loader = DataLoader(self.train_data, batch_size=int(self.args.batch_size), num_workers=int(self.args.n_threads), shuffle=True)
        self.valid_loader = DataLoader(self.valid_data, batch_size=int(self.args.batch_size), num_workers=int(self.args.n_threads), shuffle=False)
        os.makedirs(os.path.join(self.args.result_path, 'tensorboard'), exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.args.result_path, 'tensorboard'))
        self.writer.add_text('Model', str(self.model), 0)
        self.x_axis_data = [i for i in range(self.args.epochs)]
        self.test_mae = []
        self.test_loss = []
        self.test_rmse = []
        self.test_times = []            
        self.test_per_sample_times = [] 
        self.train_times = []           
        self.train_loss = []

    def train(self, epoch):
        start = time.time()

        total_loss = 0.0      # accumulated loss (sum)
        total_mae = 0.0       # accumulated |y_pred - y_true|
        total_mse = 0.0       # accumulated (y_pred - y_true)^2
        total_num = 0         # accumulated sample count
        self.model.train()
        for param_group in self.optimizer.param_groups:
            cur_lr = param_group['lr']
        start = time.time()
        num_iter = len(self.train_loader)
        total_loss, total_acc, total_num = 0., 0., 0.
        total_mse, total_cf = 0., 0.
        for i, (idx, static, nutrtion, label) in enumerate(self.train_loader):
            id = idx
            st = static.to(self.args.device)
            ni = nutrtion.to(self.args.device)
            label = label.to(self.args.device)
            output, loss, mse_loss, cf_loss = self.model(id, st, ni, label)

            # Backward and Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
             # current batch size
            batch_size = label.size(0)

            # detach to tensor, without gradient
            pred = output.detach()              # (B,)
            true = label.detach()               # (B,)

            # loss: note many loss functions are batch means
            loss_value = loss.item()
            total_loss += loss_value * batch_size   

            mse_loss_value = mse_loss.item()
            total_mse += mse_loss_value * batch_size

            cf_loss_value = cf_loss.item()
            total_cf += cf_loss_value * batch_size  

            # absolute error (MAE)
            mae_batch = torch.abs(pred - true).sum().item()
            total_mae += mae_batch

            # squared error (MSE for later RMSE)
            mse_batch = torch.pow(pred - true, 2).sum().item()
            total_mse += mse_batch

            total_num += batch_size
            self.current_step += 1

        # -------- Epoch-level metrics --------
        # average training loss
        average_loss = total_loss / total_num
        average_mse = total_mse / total_num          # average batch MSE (weighted)
        average_cf = total_cf / total_num            # average batch CF (weighted)
        mae = total_mae / total_num                  # mean absolute error
        rmse = (total_mse / total_num) ** 0.5        # root mean squared error

        train_time = time.time() - start  # epoch training time (seconds)

        # logging
        self.train_loss.append(average_loss)
        self.train_times.append(train_time)

        self.writer.add_scalar('train/train_loss', average_loss, epoch)
        self.writer.add_scalar('train/train_mae',  mae,          epoch)
        self.writer.add_scalar('train/train_rmse', rmse,         epoch)
        self.writer.add_scalar('train/train_time', train_time,   epoch)

        logging.info(
            f"Train Epoch[{epoch}] : avg_loss: {average_loss:.4f}, "
            f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, Time: {train_time:.2f}s"
            f"mse_loss: {average_mse:.4f}, cf_loss: {rmse:.4f}"
        )
        print(
            f"Train Epoch[{epoch}] : avg_loss: {average_loss:.4f}, "
)
#         print(
#             f"Train Epoch[{epoch}] : avg_loss: {average_loss:.4f}, "
#             f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, Time: {train_time:.2f}s"
# )
    def valid(self, epoch):
        self.model.eval()
        start = time.time()

        total_loss_sum = 0.0      # accumulated (batch_loss * batch_size) for average loss
        total_abs_err = 0.0       # accumulated |y_pred - y_true| for MAE
        total_sq_err = 0.0        # accumulated (y_pred - y_true)^2 for RMSE and MSE
        total_num = 0             # accumulated sample count

        all_preds = []            # store all predictions (float) for later analysis/plotting
        all_labels = []           # store all ground truth values

        with torch.no_grad():
            for i, (id, sta, ni, label) in enumerate(self.valid_loader):
                # move to device
                sta = sta.to(self.args.device)      # static features  (B, d_static)
                ni = ni.to(self.args.device)        # dynamic sequence  (B, T, F)
                label = label.to(self.args.device)  # ground truth FBG   (B,)

                output, loss_val, mse_loss_val, cf_loss_val = self.model(id, sta, ni, label)
                batch_size = label.size(0)
                # accumulate loss (multiply back by batch_size for sample-weighted average)
                total_loss_sum += loss_val.item() * batch_size

                # Error
                diff = output - label            # (B,)
                abs_err = torch.abs(diff).sum().item()
                sq_err = torch.pow(diff, 2).sum().item()

                total_abs_err += abs_err
                total_sq_err += sq_err
                total_num += batch_size

                # Save to list
                all_preds.extend(output.detach().cpu().tolist())
                all_labels.extend(label.detach().cpu().tolist())

                self.current_step += 1

        # ====== Compute epoch-level metrics ======
        test_time = time.time() - start
        per_sample_test_time = test_time / total_num if total_num > 0 else 0.0

        # average loss (sample-weighted)
        average_loss = total_loss_sum / total_num if total_num > 0 else 0.0

        # MAE: mean absolute error
        mae = total_abs_err / total_num if total_num > 0 else 0.0

        # MSE: mean squared error
        mse = total_sq_err / total_num if total_num > 0 else 0.0

        rmse = math.sqrt(total_sq_err / total_num) if total_num > 0 else 0.0

        # ====== Record to object history ======
        # using meaningful field names
        self.current_mae = mae
        self.test_mae.append(mae)
        self.test_rmse.append(rmse)
        self.test_loss.append(average_loss)
        self.test_times.append(test_time)
        self.test_per_sample_times.append(per_sample_test_time)

        # ====== Write TensorBoard / logger ======
        self.writer.add_scalar('valid/loss', average_loss, epoch)
        self.writer.add_scalar('valid/mae', mae, epoch)
        self.writer.add_scalar('valid/mse', mse, epoch)  # Add MSE to tensorboard
        self.writer.add_scalar('valid/rmse', rmse, epoch)
        self.writer.add_scalar('valid/test_time', test_time, epoch)
        self.writer.add_scalar('valid/per_sample_test_time', per_sample_test_time, epoch)

        logging.info(
            f'Valid Epoch[{epoch}] : '
            f'avg_loss: {average_loss:.4f}, '
            f'MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, '
            f'Test Time: {test_time:.2f}s, '
            f'Per-sample Test Time: {per_sample_test_time:.6f}s'
        )

        print(
            f'Valid Epoch[{epoch}] : '
            f'avg_loss: {average_loss:.4f}, '
            f'MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, '
            f'Test Time: {test_time:.2f}s, '
            f'Per-sample Test Time: {per_sample_test_time:.6f}s'
        )

        # To plot scatter (y_true vs y_pred) or compute R^2 later, use all_preds/all_labels
        # R^2 can be added here:
        y_true = np.asarray(all_labels, dtype=float)
        y_pred = np.asarray(all_preds, dtype=float)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        logging.info(
            f'Valid Epoch[{epoch}] : ... R2: {r2:.4f}'
        )
        print(
            f'Valid Epoch[{epoch}] : ... R2: {r2:.4f}'
        )
    def run(self):
        if self.args.test_only:
            self.test()
        else:
            start_epoch = 0
            best_map = 0.
            last_map = 0.
            epoch = start_epoch
            os.makedirs(os.path.join(self.args.result_path, 'checkpoint'), exist_ok=True)
            while epoch < self.args.epochs:
                self.train(epoch=epoch)
                self.valid(epoch=epoch)

                state_dict = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                epoch += 1
                if epoch % 5 == 0:
                    pth_dir = os.path.join(self.args.result_path, 'checkpoint')
                    torch.save(state_dict, f'{pth_dir}/model_epoch_{epoch}.pth')
                # if min(self.test_mae) == self.current_mae:
                #     pth_dir = os.path.join(self.args.result_path, 'checkpoint')
                #     torch.save(state_dict, f'{pth_dir}/best_model.pth')
            
            # Summarize training and testing time
            # avg_train_time = sum(self.train_times) / len(self.train_times) if self.train_times else 0
            # avg_test_time = sum(self.test_times) / len(self.test_times) if self.test_times else 0
            # avg_per_sample_test_time = sum(self.test_per_sample_times) / len(self.test_per_sample_times) if self.test_per_sample_times else 0
            # print(f'Average Training Time per Epoch: {avg_train_time:.4f}s')
            # print(f'Average Testing Time per Epoch: {avg_test_time:.4f}s')
            # print(f'Average Per-sample Testing Time: {avg_per_sample_test_time:.6f}s')
            # logging.info(f'Average Training Time per Epoch: {avg_train_time:.4f}s')
            # logging.info(f'Average Testing Time per Epoch: {avg_test_time:.4f}s')
            # logging.info(f'Average Per-sample Testing Time: {avg_per_sample_test_time:.6f}s')

            print(f'Min MAE: {min(self.test_mae):.4f}')
            logging.info(f'Min MSE: {min(self.test_mae):.4f}')
            logging.info(f'Min RMSE: {min(self.test_rmse):.4f}')

        self.writer.close()

if __name__ == '__main__':
    app = Run()
    app.run()
    print('Over')