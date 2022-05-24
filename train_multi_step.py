import torch
import numpy as np
import argparse
import time
# from util import *
from util_newdata import *
from trainer import Trainer
from net import Net
import os
from mode import Mode
import random


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:4', help='')
parser.add_argument('--data',
                    type=str,
                    default='data/METR-LA_dim4/',
                    help='data path')
parser.add_argument('--cl',
                    type=str_to_bool,
                    default=False,
                    help='whether to do curriculum learning')
parser.add_argument('--adj_data',
                    type=str,
                    default='data/METR-LA_dim4/adj_mx.pkl',
                    help='adj data path')
parser.add_argument('--num_nodes',
                    type=int,
                    default=207,
                    help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--conv_channels',
                    type=int,
                    default=32,
                    help='convolution channels')
parser.add_argument('--residual_channels',
                    type=int,
                    default=32,
                    help='residual channels')
parser.add_argument('--skip_channels',
                    type=int,
                    default=64,
                    help='skip channels')
parser.add_argument('--end_channels',
                    type=int,
                    default=128,
                    help='end channels')
parser.add_argument('--seq_in_len',
                    type=int,
                    default=12,
                    help='input sequence length')
parser.add_argument('--seq_out_len',
                    type=int,
                    default=12,
                    help='output sequence length')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='learning rate')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0001,
                    help='weight decay rate')

parser.add_argument('--step_size1', type=int, default=2500, help='step_size')
parser.add_argument('--step_size2', type=int, default=100, help='step_size')

parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=5, help='')
parser.add_argument('--seed', type=int, default=101, help='random seed')
parser.add_argument('--save', type=str, default='./save/', help='save path')
parser.add_argument('--expid', type=str, default='1', help='experiment id')

parser.add_argument('--runs', type=int, default=3, help='number of runs')
parser.add_argument('--LOAD_INITIAL',
                    default=False,
                    type=str_to_bool,
                    help='If LOAD_INITIAL.')
parser.add_argument('--TEST_ONLY',
                    default=False,
                    type=str_to_bool,
                    help='If TEST_ONLY.')
parser.add_argument('--epoch_pretest',
                    type=str,
                    default='-1',
                    help='epoch of pretest')
parser.add_argument('--tolerance',
                    type=int,
                    default=20,
                    help='tolerance for earlystopping')
parser.add_argument('--OUTPUT_PREDICTION',
                    default=False,
                    type=str_to_bool,
                    help='If OUTPUT_PREDICTION.')
parser.add_argument('--IF_PERM',
                    default=False,
                    type=str_to_bool,
                    help='IF_PERM.')
parser.add_argument('--cl_decay_steps',
                    default=2000,
                    type=float,
                    help='cl_decay_steps.')
parser.add_argument('--new_training_method',
                    default=False,
                    type=str_to_bool,
                    help='new_training_method.')

parser.add_argument('--time_slot',
                    type=int,
                    default=5,
                    help='a time step is 5 mins')
parser.add_argument('--decay_epoch', type=int, default=5, help='decay epoch')

parser.add_argument(
    '--SE_file',
    default='data/SE(METR).txt',
    help='spatial emebdding file')
parser.add_argument(
    '--adj_dtw_data',
    type=str,
    default='/data/lifuxian/lfx_STS/my_model/STS1.0/data/adj_METR-LA_003.csv',
    help='adj data path')
parser.add_argument('--forcp',
                    type=int,
                    default=0,
                    help='number for checkpoint')
parser.add_argument('--sts_kernal_size',
                    type=int,
                    default=4,
                    help='sts_kernal_size')
parser.add_argument('--lr_decay_rate',
                    type=float,
                    default=0.97,
                    help='lr_decay_rate')
parser.add_argument('--clip', type=int, default=3, help='clip')
parser.add_argument('--in_dim', type=int, default=4, help='inputs dimension')
parser.add_argument('--max_value',
                    type=float,
                    default=70.0,
                    help='70.0 for METR, 100.0 for BJ')
parser.add_argument('--layers', type=int, default=3, help='number of layers')
parser.add_argument('--gcn_depth',
                    type=int,
                    default=2,
                    help='graph convolution depth')

args = parser.parse_args()

sts_kernal_size = args.sts_kernal_size
max_value = args.max_value

T = 24 * 60 // args.time_slot
bn_decay = 0.1

num_train = 23974
lr_decay_steps = args.decay_epoch * num_train // args.batch_size

device = torch.device(args.device)

os.makedirs(args.save, exist_ok=True)

epoch_pretest = args.epoch_pretest.split('_')

dataloader, adj, adj_dtw, config, num_nodes = load_dataset(args.data, args.batch_size, 
                                                           args.batch_size, args.batch_size)
scaler = dataloader['scaler']

pre_mask = None
pre_tg1 = torch.FloatTensor(construct_tg1(adj, adj_dtw)).to(device)
pre_tg2 = torch.FloatTensor(construct_tg2(adj, adj_dtw)).to(device)
pre_sg1 = torch.FloatTensor(construct_sg1(adj, adj_dtw)).to(device)
pre_sg2 = torch.FloatTensor(construct_sg2(adj, adj_dtw)).to(device)
pre_tc = torch.FloatTensor(construct_tc(adj, adj_dtw)).to(device)
pre_tg_sg1 = torch.FloatTensor(construct_tg_sg1(adj, adj_dtw)).to(device)
pre_tg_sg2 = torch.FloatTensor(construct_tg_sg2(adj, adj_dtw)).to(device)

pre_adj_first = [pre_tg1, pre_sg1, pre_tg_sg1, pre_tg_sg2]
pre_adj_second = [pre_tg2, pre_sg2, pre_tc]

def main(runid):

    seed = runid
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False

    torch.backends.cudnn.benchmark = True

    model = Net(T,
                bn_decay,
                args.gcn_depth,
                num_nodes,
                device,
                pre_adj_first=pre_adj_first,
                pre_adj_second=pre_adj_second,
                num_of_hop=args.gcn_depth,
                use_mask = False,
                dropout=args.dropout,
                node_dim=args.node_dim,
                conv_channels=args.conv_channels,
                residual_channels=args.residual_channels,
                skip_channels=args.skip_channels,
                end_channels=args.end_channels,
                seq_length=args.seq_in_len,
                in_dim=args.in_dim,
                out_dim=args.seq_out_len,
                layers=args.layers,
                forcp=args.forcp)

    print(args)

    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip,
                     args.step_size1, args.seq_out_len, scaler, device,
                     args.new_training_method, lr_decay_steps,
                     args.lr_decay_rate, max_value)
    if args.LOAD_INITIAL:
        engine.model.load_state_dict(
            torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) +
                       ".pth",
                       map_location='cpu'))
        print('model load success!')

    if args.TEST_ONLY:

        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]

        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            with torch.no_grad():
                engine.model.eval()
                preds = engine.model(testx)
                preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze(dim=1))

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        if args.OUTPUT_PREDICTION:
            pred_all = scaler.inverse_transform(yhat).cpu()
            path_savepred = args.save + 'result_pred/' + "exp" + str(
                args.expid) + "_" + str(runid)
            os.makedirs(args.save + 'result_pred/', exist_ok=True)
            np.save(path_savepred, pred_all)
            print('result of prediction has been saved, path: ' + os.getcwd() +
                  path_savepred[1:] + '.npy' + ", shape: " +
                  str(pred_all.shape))

        mae = []
        mape = []
        rmse = []
        pred = scaler.inverse_transform(yhat)
        pred = torch.clamp(pred, min=0., max=max_value)
        tmae, tmape, trmse = metric(pred, realy)
        for i in [2, 5, 8, 11]:

            pred_singlestep = pred[:, :, i]
            real = realy[:, :, i]
            metrics = metric(pred_singlestep, real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
            mae.append(metrics[0])
            mape.append(metrics[1])
            rmse.append(metrics[2])
        return tmae, tmape, trmse, mae, mape, rmse, tmae, tmape, trmse

    else:
        print("start training...", flush=True)
        his_loss = []
        val_time = []
        train_time = []
        minl = 1e5
        minl_test = 1e5
        epoch_best = -1
        tolerance = args.tolerance
        count_lfx = 0
        batches_seen = 0
        for i in range(1, args.epochs + 1):
            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            dataloader['train_loader'].shuffle()
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                batches_seen += 1
                trainx = torch.Tensor(x).to(device)
                trainx = trainx.transpose(1, 3)
                trainy = torch.Tensor(y).to(device)
                trainy = trainy.transpose(1, 3)

                metrics = engine.train_weight(trainx, trainy[:, 0, :, :])
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])

            engine.weight_scheduler.step()

            t2 = time.time()
            train_time.append(t2 - t1)

            valid_loss = []
            valid_mape = []
            valid_rmse = []

            s1 = time.time()
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                valx = torch.Tensor(x).to(device)
                valx = valx.transpose(1, 3)
                valy = torch.Tensor(y).to(device)
                valy = valy.transpose(1, 3)
                
                metrics = engine.train_arch(valx, valy[:, 0, :, :])
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
            
            engine.arch_scheduler.step()
            s2 = time.time()

            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)

            if (i - 1) % args.print_every == 0 or str(i) in epoch_pretest:
                log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
                print(log.format(i, (s2 - s1)))
                log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
                print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse,
                                 mvalid_loss, mvalid_mape, mvalid_rmse,
                                 (t2 - t1)),
                      flush=True)

            if mvalid_loss < minl:
                torch.save(
                    engine.model.state_dict(), args.save + "exp" +
                    str(args.expid) + "_" + str(runid) + ".pth")
                minl = mvalid_loss
                epoch_best = i
                count_lfx = 0
            else:
                count_lfx += 1
                if count_lfx > tolerance:
                    break


            if str(i) in epoch_pretest:
                num_epochs = i
                print('Pre_test on epoch {}'.format(num_epochs))
                print("Average Training Time: {:.4f} secs/epoch".format(
                    np.mean(train_time)))
                print("Average Inference Time: {:.4f} secs".format(
                    np.mean(val_time)))

                realy = torch.Tensor(dataloader['y_test']).to(device)
                realy = realy.transpose(1, 3)[:, 0, :, :]

                outputs = []

                for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
                    testx = torch.Tensor(x).to(device)
                    testx = testx.transpose(1, 3)
                    testy = torch.Tensor(y).to(device)
                    testy = testy.transpose(1, 3)
                    with torch.no_grad():
                        preds = engine.model(testx)
                        preds = preds.transpose(1, 3)
                    outputs.append(preds.squeeze(dim=1))

                yhat = torch.cat(outputs, dim=0)
                yhat = yhat[:realy.size(0), ...]

                for i in [2, 5, 11]:

                    pred = torch.clamp(scaler.inverse_transform(yhat[:, :, i]),
                                       min=0.,
                                       max=max_value)
                    real = realy[:, :, i]
                    metrics = metric(pred, real)
                    log = 'Evaluate model at epoch {:d} on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
                    print(
                        log.format(num_epochs, i + 1, metrics[0], metrics[1],
                                   metrics[2]))

                bestid = np.argmin(his_loss)
                engine.model.load_state_dict(
                    torch.load(args.save + "exp" + str(args.expid) + "_" +
                               str(runid) + ".pth",
                               map_location='cpu'))

                print("The valid loss on best model is {}, epoch:{}".format(
                    str(round(his_loss[bestid], 4)), epoch_best))

                outputs = []

                for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
                    testx = torch.Tensor(x).to(device)
                    testx = testx.transpose(1, 3)
                    testy = torch.Tensor(y).to(device)
                    testy = testy.transpose(1, 3)
                    with torch.no_grad():
                        preds = engine.model(testx)
                        preds = preds.transpose(1, 3)
                    outputs.append(preds.squeeze(dim=1))

                yhat = torch.cat(outputs, dim=0)
                yhat = yhat[:realy.size(0), ...]

                for i in [2, 5, 11]:
                    pred = scaler.inverse_transform(yhat[:, :, i])
                    pred = torch.clamp(scaler.inverse_transform(yhat[:, :, i]),
                                       min=0.,
                                       max=max_value)
                    real = realy[:, :, i]
                    metrics = metric(pred, real)
                    log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
                    print(log.format(i + 1, metrics[0], metrics[1],
                                     metrics[2]))

        print("Average Training Time: {:.4f} secs/epoch".format(
            np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        bestid = np.argmin(his_loss)
        engine.model.load_state_dict(
            torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) +
                       ".pth",
                       map_location='cpu'))

        print("Training finished")
        print("The valid loss on best model is {}, epoch:{}".format(
            str(round(his_loss[bestid], 4)), epoch_best))

        outputs = []
        realy = torch.Tensor(dataloader['y_val']).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]

        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            with torch.no_grad():
                preds = engine.model(testx)
                preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze(dim=1))

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        pred = scaler.inverse_transform(yhat)
        pred = torch.clamp(pred, min=0., max=max_value)
        vmae, vmape, vrmse = metric(pred, realy)

        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]

        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            with torch.no_grad():
                preds = engine.model(testx)
                preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze(dim=1))

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        mae = []
        mape = []
        rmse = []

        pred = scaler.inverse_transform(yhat)
        pred = torch.clamp(pred, min=0., max=max_value)
        tmae, tmape, trmse = metric(pred, realy)
        for i in [2, 5, 8, 11]:

            pred_singlestep = pred[:, :, i]
            real = realy[:, :, i]
            metrics = metric(pred_singlestep, real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
            mae.append(metrics[0])
            mape.append(metrics[1])
            rmse.append(metrics[2])
        return vmae, vmape, vrmse, mae, mape, rmse, tmae, tmape, trmse


if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    tmae = []
    tmape = []
    trmse = []
    for i in range(args.runs):

        if args.runs == 1:
            i = 2
        elif args.runs == 2:
            i += 1

        i += 3

        vm1, vm2, vm3, m1, m2, m3, tm1, tm2, tm3 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)
        tmae.append(tm1)
        tmape.append(tm2)
        trmse.append(tm3)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae, 0)
    amape = np.mean(mape, 0)
    armse = np.mean(rmse, 0)

    smae = np.std(mae, 0)
    smape = np.std(mape, 0)
    srmse = np.std(rmse, 0)

    print('\n\nResults for 3 runs\n\n')

    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae), np.mean(vrmse), np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae), np.std(vrmse), np.std(vmape)))
    print('\n\n')

    print('test\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(tmae), np.mean(trmse), np.mean(tmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(tmae), np.std(trmse), np.std(tmape)))
    print('\n\n')
    print(
        'test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std'
    )
    for i in range(4):
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(
            log.format([3, 6, 9, 12][i], amae[i], armse[i], amape[i], smae[i],
                       srmse[i], smape[i]))
