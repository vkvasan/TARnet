# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:09:45 2021

@author: Ranak Roy Chowdhury
"""
import utils, argparse, warnings, sys, shutil, torch, os, numpy as np, math
warnings.filterwarnings("ignore")
import multitask_transformer_class
import multitask_transformer_class
import numpy as np
import utils
import torch
import torch.nn as nn
import time
import multitask_transformer_class
import torch
import math
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AF')
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--nlayers', type=int, default=4)
parser.add_argument('--emb_size', type=int, default=256)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--task_rate', type=float, default=0.5)
parser.add_argument('--masking_ratio', type=float, default=0.15)
parser.add_argument('--lamb', type=float, default=0.8)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--ratio_highest_attention', type=float, default=0.5)
parser.add_argument('--avg', type=str, default='macro')
parser.add_argument('--dropout', type=float, default=0.01)
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--nhid_task', type=int, default=128)
parser.add_argument('--nhid_tar', type=int, default=128)
parser.add_argument('--task_type', type=str, default='classification', help='[classification, regression]')
args = parser.parse_args()



prop = utils.get_prop(args)
path = './data/' + prop['dataset'] + '/'
    
print('Data loading start...')
    #X_train, y_train, X_test, y_test = utils.data_loader(args.dataset, path, prop['task_type'])
X_train = np.load('./easy_imu_phone/x_train.npy')
y_train = np.load('./easy_imu_phone/y_train.npy')
X_test = np.load('./easy_imu_phone/x_test.npy')
y_test = np.load('./easy_imu_phone/y_test.npy')

print('Data loading complete...')
print( X_train.shape)
print( y_train.shape)
print( X_test.shape)
print( y_test.shape)



print('Data preprocessing start...')
X_train_task, y_train_task, X_test, y_test = utils.preprocess(prop, X_train, y_train, X_test, y_test)
print(X_train_task.shape, y_train_task.shape, X_test.shape, y_test.shape)
print('Data preprocessing complete...')

prop['nclasses'] = torch.max(y_train_task).item() + 1 if prop['task_type'] == 'classification' else None
prop['dataset'], prop['seq_len'], prop['input_size'] = prop['dataset'], X_train_task.shape[1], X_train_task.shape[2]
prop['device'] = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
print('Initializing model...')
model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer = utils.initialize_training(prop)
print('Model intialized...')

    #print('Training start...')
    #utils.training(model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer, X_train_task, y_train_task, X_test, y_test, prop)
    #print('Training complete...')
    #torch.save(model,'train_model.pth')
    
    #new_model =  multitask_transformer_class.MultitaskTransformerModel(prop['task_type'], prop['device'], prop['nclasses'], prop['seq_len'], prop['batch'], \
    #    prop['input_size'], prop['emb_size'], prop['nhead'], prop['nhid'], prop['nhid_tar'], prop['nhid_task'], prop['nlayers'], prop['dropout']).to(prop['device'])
    
    #new_model = torch.load('train_model.pth')
    #new_model.load_state_dict(torch.load('train_model.pth'))    
    #$print('Testing started')
    #results = utils.test(new_model, X_test, y_test, prop['batch'], prop['nclasses'], criterion_task, prop['task_type'] , prop['device'], prop['avg'])
    #print(results)
    #print('Testing complete...')

    #model, X_test, y_test, batch, nclasses, criterion_task, task_type, device, avg
    #model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer, X_train_task, y_train_task, X_test, y_test, prop):

    
class Config:
        def __init__(self):
            self.dataset = 'AF'
            self.batch = 64
            self.lr = 0.001
            self.nlayers = 4
            self.emb_size = 256
            self.nhead = 8
            self.task_rate = 0.5
            self.masking_ratio = 0.15
            self.lamb = 0.8
            self.epochs = 50
            self.ratio_highest_attention = 0.5
            self.avg = 'macro'
            self.dropout = 0.01
            self.nhid = 128
            self.nhid_task = 128
            self.nhid_tar = 128
            self.task_type = 'classification' #if 'classification' else 0  # Converts to 1 for classification and 0 for regression        
            self.device = "cpu"
            self.nclasses = None
            self.seq_len = 0 
            self.input_size = 0
            #self.nclasses = torch.max(y_train_task).item() + 1 if prop['task_type'] == 'classification' else None
            #self.dataset = dataset

        def __repr__(self):
            return (f"Config(dataset={self.dataset}, batch={self.batch}, lr={self.lr}, nlayers={self.nlayers}, "
                    f"emb_size={self.emb_size}, nhead={self.nhead}, task_rate={self.task_rate}, "
                    f"masking_ratio={self.masking_ratio}, lamb={self.lamb}, epochs={self.epochs}, "
                    f"ratio_highest_attention={self.ratio_highest_attention}, avg={self.avg}, "
                    f"dropout={self.dropout}, nhid={self.nhid}, nhid_task={self.nhid_task}, "
                    f"nhid_tar={self.nhid_tar}, task_type={self.task_type})")


        # Instantiate the class
config = Config()

X_train = np.load('./easy_imu_phone/x_train.npy')
y_train = np.load('./easy_imu_phone/y_train.npy')
X_test = np.load('./easy_imu_phone/x_test.npy')
y_test = np.load('./easy_imu_phone/y_test.npy')
criterion_task = torch.nn.CrossEntropyLoss()

print('Data loading complete...')
print( X_train.shape)
print( y_train.shape)
print( X_test.shape)
print( y_test.shape)



print('Data preprocessing start...')
X_train_task, y_train_task, X_test, y_test = utils.preprocess_nb(config, X_train, y_train, X_test, y_test)
print(X_train_task.shape, y_train_task.shape, X_test.shape, y_test.shape)
print('Data preprocessing complete...')

config.nclasses = torch.max(y_train_task).item() + 1 if config.task_type == 'classification' else None
config.seq_len, config.input_size = X_train_task.shape[1], X_train_task.shape[2]
#prop['device'] = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
            

        # Access the arguments as attributes
print(config)

import torch
import torch.nn as nn
import math
import transformer

class PositionalEncoding(nn.Module):
            def __init__(self, seq_len, d_model, dropout=0.1):
                super(PositionalEncoding, self).__init__()
                max_len = max(5000, seq_len)
                self.dropout = nn.Dropout(p=dropout)
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                
                if d_model % 2 == 0:
                    pe[:, 1::2] = torch.cos(position * div_term)
                else:
                    pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
                
                pe = pe.unsqueeze(0).transpose(0, 1)
                self.register_buffer('pe', pe)

            def forward(self, x):
                x = x + self.pe[:x.size(0), :]
                return self.dropout(x)

class Permute(nn.Module):
            def forward(self, x):
                return x.permute(1, 0)

class MultitaskTransformerModel(nn.Module):
            def __init__(self, task_type, device, nclasses, seq_len, batch, input_size, emb_size, nhead, nhid, nhid_tar, nhid_task, nlayers, dropout=0.1):
                super(MultitaskTransformerModel, self).__init__()
                self.device = device

                self.trunk_net = nn.Sequential(
                    nn.Linear(input_size, emb_size).to(device),
                    nn.BatchNorm1d(batch).to(device),
                    PositionalEncoding(seq_len, emb_size, dropout).to(device),
                    nn.BatchNorm1d(batch).to(device)
                )
                
                encoder_layers = transformer.TransformerEncoderLayer(emb_size, nhead, nhid, dropout).to(device)
                self.transformer_encoder = transformer.TransformerEncoder(encoder_layers, nlayers, device).to(device)
                
                self.batch_norm = nn.BatchNorm1d(batch).to(device)
                
                # Task-aware Reconstruction Layers
                self.tar_net = nn.Sequential(
                    nn.Linear(emb_size, nhid_tar).to(device),
                    nn.BatchNorm1d(batch).to(device),
                    nn.Linear(nhid_tar, nhid_tar).to(device),
                    nn.BatchNorm1d(batch).to(device),
                    nn.Linear(nhid_tar, input_size).to(device),
                )

                if task_type == 'classification':
                    # Classification Layers
                    self.class_net = nn.Sequential(
                        nn.Linear(emb_size, nhid_task).to(device),
                        nn.ReLU().to(device),
                        Permute().to(device),
                        nn.BatchNorm1d(batch).to(device),
                        Permute().to(device),
                        nn.Dropout(p=0.3).to(device),
                        nn.Linear(nhid_task, nhid_task).to(device),
                        nn.ReLU().to(device),
                        Permute().to(device),
                        nn.BatchNorm1d(batch).to(device),
                        Permute().to(device),
                        nn.Dropout(p=0.3).to(device),
                        nn.Linear(nhid_task, nclasses).to(device)
                    )
                else:
                    # Regression Layers
                    self.reg_net = nn.Sequential(
                        nn.Linear(emb_size, nhid_task).to(device),
                        nn.ReLU().to(device),
                        Permute().to(device),
                        nn.BatchNorm1d(batch).to(device),
                        Permute().to(device),
                        nn.Linear(nhid_task, nhid_task).to(device),
                        nn.ReLU().to(device),
                        Permute().to(device),
                        nn.BatchNorm1d(batch).to(device),
                        Permute().to(device),
                        nn.Linear(nhid_task, 1).to(device),
                    )

            def forward(self, x, task_type):
                #x = x.to(self.device)
                #print(f"Input x is on device: {x.device}")
                
                x = self.trunk_net(x.permute(1, 0, 2))
                #print(f"x after trunk_net is on device: {x.device}")
                
                x, attn = self.transformer_encoder(x)
                #print(f"x after transformer_encoder is on device: {x.device}")
                #print(f"attn after transformer_encoder is on device: {attn.device}")
                
                x = self.batch_norm(x)
                #print(f"x after batch_norm is on device: {x.device}")
                
                if task_type == 'reconstruction':
                    output = self.tar_net(x).permute(1, 0, 2)
                elif task_type == 'classification':
                    output = self.class_net(x[-1])
                elif task_type == 'regression':
                    output = self.reg_net(x[-1])
                
                return output, attn
            
            # Load the entire model and ensure it is on the correct device
device = torch.device('cpu')  # or 'cuda:0' if using GPU
old_model = torch.load('train_model.pth', map_location=device)
state_dict = old_model.state_dict()

model =  MultitaskTransformerModel(config.task_type, config.device, config.nclasses, config.seq_len, config.batch, \
config.input_size, config.emb_size, config.nhead, config.nhid, config.nhid_tar, config.nhid_task, config.nlayers, config.dropout)
model.load_state_dict(state_dict)



def evaluate(y_pred, y, nclasses, criterion, task_type, device, avg):
                results = []

                if task_type == 'classification':
                    loss = criterion(y_pred.view(-1, nclasses), torch.as_tensor(y, device=device)).item()
                    
                    pred, target = y_pred.cpu().data.numpy(), y.cpu().data.numpy()
                    pred = np.argmax(pred, axis=1)
                    acc = accuracy_score(target, pred)
                    prec = precision_score(target, pred, average=avg)
                    rec = recall_score(target, pred, average=avg)
                    f1 = f1_score(target, pred, average=avg)
                    
                    results.extend([loss, acc, prec, rec, f1])
                else:
                    y_pred = y_pred.squeeze()
                    y = torch.as_tensor(y, device=device)
                    rmse = math.sqrt(((y_pred - y) ** 2).sum().item() / y_pred.shape[0])
                    mae = torch.abs(y_pred - y).mean().item()
                    results.extend([rmse, mae])
                
                return results

def test_with_inference_time(model, X, y, batch, nclasses, criterion, task_type, device, avg):
                model.eval()  # Turn on the evaluation mode
                model.to(device)  # Ensure the entire model is on the correct device

                num_batches = math.ceil(X.shape[0] / batch)
                output_arr = []
                total_inference_time = 0.0  # Initialize the total inference time

                with torch.no_grad():
                    for i in range(num_batches):
                        start = int(i * batch)
                        end = int((i + 1) * batch)
                        num_inst = y[start:end].shape[0]

                        # Ensure X_batch is on the correct device
                        X_batch = torch.as_tensor(X[start:end], device=device)
                        
                        # Start timing before the model makes predictions
                        start_time = time.time()
                        out = model(X_batch, task_type)[0]
                        end_time = time.time()
                        
                        # Calculate the time taken for this batch and add it to the total
                        batch_inference_time = end_time - start_time
                        total_inference_time += batch_inference_time
                        
                        output_arr.append(out[:num_inst])

                # Calculate average inference time per batch
                avg_inference_time_per_batch = total_inference_time / num_batches
                
                # Calculate average inference time per sample
                avg_inference_time_per_sample = total_inference_time / (num_batches * batch)
                
                # Evaluate the model predictions
                results = evaluate(torch.cat(output_arr, 0), y, nclasses, criterion, task_type, device, avg)
                
                return results, avg_inference_time_per_batch, avg_inference_time_per_sample

            # Example of calling the function
results, avg_inference_time_per_batch, avg_inference_time_per_sample = test_with_inference_time(
                model, X_test, y_test, config.batch, config.nclasses, criterion_task, config.task_type, device, config.avg
)
print("Evaluation Results:", results)
print("Average Inference Time per Batch:", avg_inference_time_per_batch, "seconds")
print("Average Inference Time per Sample:", avg_inference_time_per_sample, "seconds")


