# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:09:45 2021

@author: Ranak Roy Chowdhury
"""
import utils, argparse, warnings, sys, shutil, torch, os, numpy as np, math
warnings.filterwarnings("ignore")
import multitask_transformer_class



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



def main():    
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

    print('Training start...')
    utils.training(model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer, X_train_task, y_train_task, X_test, y_test, prop)
    print('Training complete...')
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

if __name__ == "__main__":
    main()