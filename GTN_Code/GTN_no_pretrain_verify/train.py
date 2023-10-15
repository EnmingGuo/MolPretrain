import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, recall_score
from dataset import MolTestDatasetWrapper
from gtn import GTN
import argparse
import matplotlib.pyplot as plt
# import wandb

import warnings

warnings.filterwarnings("ignore") # gx

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class Model(nn.Module):   
    def __init__(self, GTN, task, node_dim, finetune=False):
        super(Model, self).__init__()
        self.gnn = GTN
        self.task = task
        self.node_dim = node_dim
        self.finetune = finetune

        if self.finetune:
            for _, parms in gnn.named_parameters():
                parms.requires_grad = False

        if self.task == 'classification':
            self.pred_head = nn.Sequential(
                    nn.Linear(self.node_dim, self.node_dim//2), 
                    nn.Softplus(),
                    nn.Linear(self.node_dim//2, 2),
                )

        elif task == 'regression':
            self.pred_head = nn.Sequential(
                nn.Linear(self.node_dim, self.node_dim//2), 
                nn.Softplus(),
                nn.Linear(self.node_dim//2, 1)
            )
    
    def forward(self, data):
        gnn_output = self.gnn(data)
        return self.pred_head(gnn_output)


def init_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    init_seed(777)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Training Epochs')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='number of workers')
    parser.add_argument('--dataset', type=str, default='BBBP',
                        help='dataset path')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='hidden dimensions')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--warm_up', type=int, default=20,
                        help='warm up rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GT/FastGT layers')
    parser.add_argument("--channel_agg", type=str, default='concat')
    parser.add_argument("--remove_self_loops", action='store_true', help="remove_self_loops")

    args = parser.parse_args()
    print(args)

    epochs = args.epoch
    batch_size = args.batch
    num_workers = args.num_workers
    dataset = args.dataset
    node_dim = args.node_dim
    emb_dim = args.emb_dim
    num_channels = args.num_channels
    init_lr = args.lr
    warm_up = args.warm_up
    weight_decay = args.weight_decay
    num_layers = args.num_layers

    # wandb.login(key='ff36dda227a04150a0cacc715b2460176efe3144')
    # wandb.init(
    #     project='GTN Molecule',
    #     name = 'deepchem_test_30_features_lr_1e-5'
    # )

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # datasets = ['BBBP', 'Tox21', 'ClinTox', 'HIV', 'BACE', 'SIDER', 'MUV', 'FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']

    if dataset == 'BBBP':
        task = 'classification'
        task_name = 'BBBP'
        path = '/data1/gx/GTN_Code/dataset/BBBP.csv'
        target_list = ["p_np"]

    elif dataset == 'Tox21':
        task = 'classification'
        task_name = 'Tox21'
        path = '../dataset/tox21.csv'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif dataset == 'ClinTox':
        task = 'classification'
        task_name = 'ClinTox'
        path = '../dataset/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif dataset == 'HIV':
        task = 'classification'
        task_name = 'HIV'
        path = '../dataset/HIV.csv'
        target_list = ["HIV_active"]

    elif dataset == 'BACE':
        task = 'classification'
        task_name = 'BACE'
        path = '../dataset/bace.csv'
        target_list = ["Class"]

    elif dataset == 'SIDER':
        task = 'classification'
        task_name = 'SIDER'
        path = '../dataset/sider.csv'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]

    elif dataset == 'MUV':
        task = 'classification'
        task_name = 'MUV'
        path = '../dataset/muv.csv'
        target_list = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]

    elif dataset == 'FreeSolv':
        task = 'regression'
        task_name = 'FreeSolv'
        path = '../dataset/freesolv.csv'
        target_list = ["expt"]

    elif dataset == 'ESOL':
        task = 'regression'
        task_name = 'ESOL'
        path = '../dataset/esol.csv'
        target_list = ["measured log solubility in mols per litre"]

    elif dataset == 'Lipo':
        task = 'regression'
        task_name = 'Lipo'
        path = '../dataset/Lipophilicity.csv'
        target_list = ["exp"]

    elif dataset == 'qm7':
        task = 'regression'
        task_name = 'qm7'
        path = '../dataset/qm7.csv'
        target_list = ["u0_atom"]

    elif dataset == 'qm8':
        task = 'regression'
        task_name = 'qm8'
        path = '../dataset/qm8.csv'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
            "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]

    elif dataset == 'qm9':
        task = 'regression'
        task_name = 'qm9'
        path = '../dataset/qm9.csv'
        target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']

    else:
        raise ValueError('Undefined downstream task!')

    if task == 'classification':
        loss_func = nn.CrossEntropyLoss()
    elif task == 'regression':
        if task_name in ['qm7', 'qm8', 'qm9']:
            loss_func = nn.L1Loss()
        else:
            loss_func = nn.MSELoss()

    gnn = GTN(
        task = task,
        num_channels = num_channels,
        w_in = 30,
        w_out = node_dim,
        num_layers = num_layers,
        emb_dim = emb_dim,
        args = args
    )
    # model.load_state_dict(torch.load('best_model_params.pth'))

    model = Model(gnn, task, node_dim, finetune=False)

    model.to(device)

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr,
        weight_decay=weight_decay
    )

    scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs-warm_up, 
            eta_min=0, last_epoch=-1
    )

    # wandb.watch(model)
    for target in target_list:
        dataset = MolTestDatasetWrapper(batch_size=batch_size, num_workers=0, valid_size=0.1, test_size=0.1, data_path=path, target=target,task=task, splitting='random')
        train_loader, valid_loader, test_loader = dataset.get_data_loaders()

        print(f'Working on dataset {task_name} with target {target}')

        if task_name in ['qm7', 'qm9']:
            labels = []
            for _ , d in enumerate(train_loader):
                labels.append(d['atom'].y)
            labels = torch.cat(labels)
            normalizer = Normalizer(labels)
            print(normalizer.mean, normalizer.std, labels.shape)
        
        else:
            normalizer = None

        val_loss = []
        val_auc = []
        val_recall = []
        val_mae = []
        val_rmse = []
        train_loss = []
        train_batch_loss = []
        best_valid_loss = 999.0
        # 训练
        for epoch in range(epochs):
            total_loss = 0
            # with torch.autograd.detect_anomaly():
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(device)
                pred = model(data)

                if task == 'classification':
                    loss = loss_func(pred, data['atom'].y.flatten())
                elif task == 'regression':
                    if normalizer:
                        loss = loss_func(pred, normalizer.norm(data['atom'].y))
                    else:
                        loss = loss_func(pred, data['atom'].y)
                # wandb.log({'batch_loss':loss.item()})

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                train_batch_loss.append(loss.item())

            avg_train_loss = total_loss / len(train_loader)
            train_loss.append(avg_train_loss)
            # wandb.log({'metric':avg_train_loss,'lr':optimizer.param_groups[0]['lr']})

            # warm up
            if epoch >= warm_up:
                scheduler.step()

            # 验证
            if (epoch + 1) % 5 == 0:
                predictions = []
                labels = []
                with torch.no_grad():
                    model.eval()
                    valid_loss = 0.0
                    num_data = 0

                    for bn, data in enumerate(valid_loader):
                        data = data.to(device)

                        pred = model(data) # 这里的值是nan

                        if task == 'classification':
                            loss = loss_func(pred, data['atom'].y.flatten())
                        elif task == 'regression':
                            if normalizer:
                                loss = loss_func(pred, normalizer.norm(data['atom'].y))
                            else:
                                loss = loss_func(pred, data['atom'].y)

                        valid_loss += loss.item() * data['atom'].y.size(0)
                        num_data += data['atom'].y.size(0)

                        if normalizer:
                            pred = normalizer.denorm(pred)

                        if task == 'classification':
                            pred = F.softmax(pred, dim=-1)

                        predictions.extend(pred.cpu().detach().numpy())
                        labels.extend(data['atom'].y.cpu().flatten().numpy())

                    valid_loss /= num_data

                model.train()
                if task == 'regression':
                    predictions = np.array(predictions)
                    labels = np.array(labels)
                    if task_name in ['qm7', 'qm8', 'qm9']:
                        mae = mean_absolute_error(labels, predictions)
                        print('Validation loss:', valid_loss, 'MAE:', mae)

                        val_loss.append(valid_loss)
                        val_mae.append(mae)
                    else:
                        try:
                            rmse = mean_squared_error(labels, predictions, squared=False)
                        except Exception:
                            import ipdb
                            ipdb.set_trace()
                        print('Validation loss:', valid_loss, 'RMSE:', rmse)

                        val_loss.append(valid_loss)
                        val_rmse.append(rmse)

                elif task == 'classification':
                    predictions = np.array(predictions)
                    labels = np.array(labels)
                    try:
                        roc_auc = roc_auc_score(labels, predictions[:,1])
                        # recall = recall_score(labels, predictions[:,1])
                    except ValueError as e:
                        if "Only one class present in y_true" in str(e):
                            continue
                        else:
                            print('ROC AUC computing error!')
                            import ipdb
                            ipdb.set_trace()
                    print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)

                    val_loss.append(valid_loss)
                    val_auc.append(roc_auc)
                    # val_recall.append(recall)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), f'./{task_name}/model_params.pth')

            if (epoch+1) % 100 == 0: # save checkpoint
                torch.save(model.state_dict(), f'./{task_name}/model_{epoch+1}.pth')
            
        plt.figure()
        plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training Loss on task {task_name}')
        plt.legend()
        plt.savefig(f'./{task_name}/train_loss_avg_{target}.png')

        plt.figure()
        plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Validation Loss on task {task_name}')
        plt.legend()
        plt.savefig(f'./{task_name}/val_loss_{target}.png')      

        plt.figure()
        plt.plot(range(len(train_batch_loss)), train_batch_loss, label='Batch Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Batch Loss on task {task_name}')
        plt.legend()
        plt.savefig(f'./{task_name}/train_batch_loss_{target}.png')

        if task == 'regression':
            if task_name in ['qm7', 'qm8', 'qm9']:
                plt.figure()
                plt.plot(range(len(val_mae)), val_mae, label='MAE')
                plt.xlabel('Epochs')
                plt.ylabel('MAE')
                plt.title(f'MAE on target {target}')
                plt.legend()
                plt.savefig(f'./{task_name}/val_mae_{target}.png')
            
            else:
                plt.figure()
                plt.plot(range(len(val_rmse)), val_rmse, label='RMSE')
                plt.xlabel('Epochs')
                plt.ylabel('RMSE')
                plt.title(f'RMSE with target {target}')
                plt.legend()
                plt.savefig(f'./{task_name}/val_rmse_{target}.png')

        elif task == 'classification':
            plt.figure()
            plt.plot(range(len(val_auc)), val_auc, label='ROC_AUC')
            plt.xlabel('Epochs')
            plt.ylabel('AUC')
            plt.title(f'ROC_AUC with target {target}')
            plt.legend()
            plt.savefig(f'./{task_name}/val_roc_auc_{target}.png')

        # 测试
        model.load_state_dict(torch.load(f'./{task_name}/model_params.pth'))
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(device)

                pred = model(data)

                if task == 'classification':
                    loss = loss_func(pred, data['atom'].y.flatten())
                elif task == 'regression':
                    if normalizer:
                        loss = loss_func(pred, normalizer.norm(data['atom'].y))
                    else:
                        loss = loss_func(pred, data['atom'].y)

                test_loss += loss.item() * data['atom'].y.size(0)
                num_data += data['atom'].y.size(0)

                if normalizer:
                    pred = normalizer.denorm(pred)

                if task == 'classification':
                    pred = F.softmax(pred, dim=-1)
                predictions.extend(pred.cpu().detach().numpy())
                labels.extend(data['atom'].y.cpu().flatten().numpy())

            test_loss /= num_data
        
        model.train()

        if task == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if task_name in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print(f'Test loss on task {task_name}:', test_loss, 'Test MAE:', mae)
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print(f'Test loss on task {task_name}:', test_loss, 'Test RMSE:', rmse)

        elif task == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            print(f'Test loss on task {task_name}:', test_loss, 'Test ROC AUC:', roc_auc)
        
        result = {'task_name': [], 'metric': [], 'epoch':epochs}
        result['task_name'].append((task_name, target))
        if task == 'regression':
            if task_name in ['qm7', 'qm8', 'qm9']:
                result['metric'].append(('MAE', mae))
            else:
                result['metric'].append(('RMSE', rmse))
        elif task == 'classification':
            result['metric'].append(('ROC_AUC', roc_auc))


        df = pd.DataFrame(result)

        csv_file = f'./{task_name}/test_results.csv'
        df.to_csv(csv_file, mode='a+', index=False)
