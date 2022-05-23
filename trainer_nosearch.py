import torch.optim as optim
import math
from net import *
import util
from mode import Mode

class Trainer():
    def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, scaler, 
                 device, cl=True, new_training_method = False, lr_decay_steps = None, 
                 lr_decay_rate = 0.97, max_value = None):
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.weight_optimizer = optim.Adam(self.model.weight_parameters(), lr=lrate, weight_decay=wdecay)
        self.arch_optimizer = optim.Adam(self.model.arch_parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.clip = clip
        self.step = step_size
         
        self.iter = 0
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl
        self.new_training_method = new_training_method
        self.max_value = max_value

        self.weight_scheduler = optim.lr_scheduler.LambdaLR(self.weight_optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)
        self.arch_scheduler = optim.lr_scheduler.LambdaLR(self.arch_optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)


    def train_weight(self, input, real_val, ycl, batches_seen = None, mode = Mode.TWO_PATHS):
        self.iter += 1  

        if self.iter%self.step==0 and self.task_level<self.seq_out_len:  
            self.task_level +=1
            if self.new_training_method:
                self.iter = 0

        self.model.train()
        self.weight_optimizer.zero_grad()
        if self.cl:
            output = self.model(input, ycl = ycl, batches_seen = self.iter, task_level = self.task_level, mode = Mode.ONE_PATH_FIXED)
        else:
            output = self.model(input, ycl = ycl, batches_seen = self.iter, task_level = self.seq_out_len, mode = Mode.ONE_PATH_FIXED)

        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        if self.cl:

            loss = self.loss(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
            mape = util.masked_mape(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0).item()
            rmse = util.masked_rmse(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0).item()
        else:
            loss = self.loss(predict, real, 0.0)
            mape = util.masked_mape(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()

        loss.backward(retain_graph=False)

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.weight_parameters(), self.clip)

        self.weight_optimizer.step()
         

        return loss.item(),mape,rmse
    
    def train_arch(self, input, real_val, ycl, mode = Mode.TWO_PATHS):
        self.model.eval()
        self.arch_optimizer.zero_grad()

        output = self.model(input, ycl = ycl, mode = Mode.TWO_PATHS)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        loss.backward(retain_graph=False)

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.arch_parameters(), self.clip)

        self.arch_optimizer.step()
         
        return loss.item(),mape,rmse    

    def eval(self, input, real_val, ycl, mode = Mode.ONE_PATH_FIXED):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input, ycl = ycl, mode = Mode.ONE_PATH_FIXED)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        predict = torch.clamp(predict, min=0., max=self.max_value)  
         
        loss = self.loss(predict, real, 0.0)  
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse



class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params   
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
         
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        self.optimizer.step()
        return  grad_norm

     
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
         
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()
