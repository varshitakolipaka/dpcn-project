import torch, torchmetrics, tqdm, copy, time
from utils import LinearLR, unlearn_func, ssd_tuning, distill_kl_loss, compute_accuracy, ssd_tuning_nc
from torch.cuda.amp import autocast
import numpy as np
from torch.cuda.amp import GradScaler
from os import makedirs
from os.path import exists
from torch.nn import functional as F
import itertools
from sklearn.metrics import classification_report
from utils import *
from torchmetrics import AUROC
from opts import parse_args
opt = parse_args()
opt.max_lr, opt.train_iters = opt.unlearn_lr, opt.unlearn_iters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_success_rate(data, model, target_label=1):
    with torch.no_grad():
        count = 0
        # for i, data in enumerate(loader):
        #     data = data.to(device)
        #     by = data.y.to(device)

        #     count += by.size(0)

        #     logits = model(data)
        #     running_acc += (torch.max(logits, dim=1)[1] == target_label).float().sum(0).cpu().numpy()
        count = data.y.size(0)
        logits = model(data)
        acc = (torch.max(logits, dim=1)[1] == target_label).float().sum(0).cpu().numpy() / count

    print(f'Trigger success rate: {acc:.4f}')

'''
high level train/eval methods
'''
class Naive():
    def __init__(self, opt, model, prenet=None):
        self.opt = opt
        opt.train_iters = opt.pretrain_iters
        self.opt.train_iters = opt.pretrain_iters
        self.curr_step, self.best_top1 = 0, 0
        self.best_model = None
        self.set_model(model, prenet)
        self.save_files = {'train_top1':[], 'val_top1':[], 'train_time_taken':0}
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.pretrain_lr, momentum=0.9, weight_decay=self.opt.wd)
        self.scheduler = LinearLR(self.optimizer, T=self.opt.train_iters*1.25, warmup_epochs=self.opt.train_iters//100) # Spend 1% time in warmup, and stop 66% of the way through training 
        if opt.task == 'gc':
            self.top1 = torchmetrics.Accuracy(task="binary", num_classes=2)
        else:
            self.top1 = torchmetrics.Accuracy(task="multiclass", num_classes=7)
        self.scaler = GradScaler()


    def set_model(self, model, prenet=None):
        self.prenet = None
        self.model = model
        self.model


    def forward_pass(self, data):
        if self.prenet is not None:
            # with torch.no_grad():
            #     feats = self.prenet(images)
            # output = self.model(feats)
            pass
        else:
            output = self.model(data)
        loss = F.cross_entropy(output, data.y)
        self.top1(output, data.y)
        return loss


    def train_one_epoch_gc(self, loader):
        self.model.train()
        self.top1.reset()

        for data in (loader):
            with autocast():
                self.optimizer.zero_grad()
                loss = self.forward_pass(data)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.curr_step += 1
                if self.curr_step > self.opt.train_iters:
                    break

        top1 = self.top1.compute().item()
        self.top1.reset()
        self.save_files['train_top1'].append(top1)
        # print(f'Step: {self.curr_step} Train Top1: {top1:.3f}')
        return
    
    def train_one_epoch(self, data, mask):
        self.model.train()
        self.top1.reset()

        with autocast():
            if self.curr_step <= self.opt.train_iters:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output[mask], data.y[mask])
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.curr_step += 1

        top1 = self.top1.compute().item()
        self.top1.reset()
        self.save_files['train_top1'].append(top1)
        # print(f'Step: {self.curr_step} Train Top1: {top1:.3f}')
        return


    def eval_gc(self, loader, save_model=True, save_preds=False):
        self.model.eval()
        self.top1.reset()

        # if save_preds:
        #     preds, targets = [], []

        with torch.no_grad():
            for data in (loader):
                with autocast():
                    output = self.model(data)
                pred = torch.argmax(output, dim=1)
                self.top1(pred, data.y)
                # if save_preds:
                #     preds.append(output)
                #     targets.append(data.y)

        top1 = self.top1.compute().item()
        self.top1.reset()
        # if not save_preds: print(f'Step: {self.curr_step} Val Top1: {top1*100:.2f}')
        
        if save_model:
            self.save_files['val_top1'].append(top1)
            if top1 > self.best_top1:
                self.best_top1 = top1
                self.best_model = copy.deepcopy(self.model).cpu()

        self.model.train()
        if save_preds:
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            return preds, targets
        return
    
    def eval(self, data, save_model=True, save_preds=False):
        self.model.eval()
        self.top1.reset()

        # if save_preds:
        #     preds, targets = [], []

        with torch.no_grad():
            with autocast():
                output = self.model(data)
            pred = torch.argmax(output, dim=1)
            self.top1(pred[data.test_mask], data.y[data.test_mask])
            # if save_preds:
            #     preds.append(output)
            #     targets.append(data.y)

        top1 = self.top1.compute().item()
        self.top1.reset()
        print("Top 1: ", top1)
        # if not save_preds: print(f'Step: {self.curr_step} Val Top1: {top1*100:.2f}')
        
        if save_model:
            self.save_files['val_top1'].append(top1)
            if top1 > self.best_top1:
                self.best_top1 = top1
                self.best_model = copy.deepcopy(self.model).cpu()

        self.model.train()
        if save_preds:
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            return preds, targets
        return


    def unlearn(self, train_loader, test_loader, eval_loaders=None):
        while self.curr_step < self.opt.train_iters:
            time_start = time.process_time()
            self.train_one_epoch(loader=train_loader)
            self.eval(test_loader)
            self.save_files['train_time_taken'] += time.process_time() - time_start
        return


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+self.opt.unlearn_method+'_'+self.opt.exp_name
        if self.opt.unlearn_method != 'Naive': 
            self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        return


    def compute_and_save_results(self, train_test_loader, test_loader, adversarial_train_loader, adversarial_test_loader):
        self.get_save_prefix()
        print(self.unlearn_file_prefix)
        if not exists(self.unlearn_file_prefix):
            makedirs(self.unlearn_file_prefix)

        torch.save(self.best_model.state_dict(), self.unlearn_file_prefix+'/model.pth')
        np.save(self.unlearn_file_prefix+'/train_top1.npy', self.save_files['train_top1'])
        np.save(self.unlearn_file_prefix+'/val_top1.npy', self.save_files['val_top1'])
        np.save(self.unlearn_file_prefix+'/unlearn_time.npy', self.save_files['train_time_taken'])
        self.model = self.best_model

        print('==> Completed! Unlearning Time: [{0:.3f}]\t'.format(self.save_files['train_time_taken']))
        
        for loader, name in [(train_test_loader, 'train'), (test_loader, 'test'), (adversarial_train_loader, 'adv_train'), (adversarial_test_loader, 'adv_test')]:
            if loader is not None:
                preds, targets = self.eval(loader=loader, save_preds=True)
                np.save(self.unlearn_file_prefix+'/preds_'+name+'.npy', preds)
                np.save(self.unlearn_file_prefix+'/targets'+name+'.npy', targets)
        return
    
class ApplyK(Naive):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
    

    def set_model(self, model, prenet):
        prenet, model = self.divide_model(model, k=self.opt.k, model_name=self.opt.model)
        model = unlearn_func(model=model, method=self.opt.unlearn_method, factor=self.opt.factor, device='cpu') 
        self.model = model
        self.prenet = prenet
        # self.model
        if self.prenet is not None: self.prenet.eval()


    def divide_model(self, model, k, model_name):
        if k == -1: # -1 means retrain all layers
            net = model
            prenet = None
            return prenet, net

        if model_name == 'resnet9':
            assert(k in [1,2,4,5,7,8])
            mapping = {1:6, 2:5, 4:4, 5:3, 7:2, 8:1}
            dividing_part = mapping[k]
            all_mods = [model.conv1, model.conv2, model.res1, model.conv3, model.res2, model.conv4, model.fc] 
            prenet = torch.nn.Sequential(*all_mods[:dividing_part])
            net = torch.nn.Sequential(*all_mods[dividing_part:])

        elif model_name == 'resnetwide28x10':
            assert(k in [1,3,5,7,9,11,13,15,17,19,21,23,25])
            all_mods = [model.conv1, model.layer1, model.layer2, model.layer3, model.norm, model.fc]
            mapping = {1:5, 9:3, 17:2, 25:1}
    
            if k in mapping:
                intervention_point = mapping[k]
                prenet = torch.nn.Sequential(*all_mods[:intervention_point])
                net = torch.nn.Sequential(*all_mods[intervention_point:])
            else:
                vals = list(mapping.keys())
                for val in vals:
                    if val > k:
                        sel_idx = val
                        break
                layer = mapping[sel_idx]
                prenet_list = all_mods[:layer]
                prenet_additions = list(all_mods[layer][:int(4-(((k-1)//2)%4))])
                prenet = torch.nn.Sequential(*(prenet_list+prenet_additions))
                net_list = list(all_mods[layer][int(4-(((k-1)//2)%4)):])
                net_additions = all_mods[layer+1:]
                net = torch.nn.Sequential(*(net_list+net_additions))

        elif model_name == 'vitb16':
            assert(k in [1,2,3,4,5,6,7,8,9,10,11,12,13])
            all_mods = [model.patch_embed, model.blocks, model.norm, model.head]
            mapping = {1:3, 13:1}

            if k in mapping:
                intervention_point = mapping[k]
                prenet = torch.nn.Sequential(*all_mods[:intervention_point])
                net = torch.nn.Sequential(*all_mods[intervention_point:])
            else:
                prenet = [model.patch_embed]
                k = 13-k
                prenet += [model.blocks[:k]]
                prenet = torch.nn.Sequential(*prenet)
                net = [model.blocks[k:], model.norm, model.head]
                net = torch.nn.Sequential(*net)
                
        prenet.to(self.opt.device)
        net.to(self.opt.device)
        return prenet, net


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)

        return 


class Scrub(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.og_model = copy.deepcopy(model)
        self.og_model.eval()
        opt.train_iters = opt.unlearn_iters
        self.opt.train_iters = opt.unlearn_iters
    

    def forward_pass(self, data):
        if self.prenet is not None:
            pass
        else:
            output = self.model(data)

        with torch.no_grad():
            logit_t = self.og_model(data)

        loss = F.cross_entropy(output, data.y)
        loss += self.opt.alpha * distill_kl_loss(output, logit_t, self.opt.kd_T)
        
        if self.maximize:
            loss = -loss
        pred = torch.argmax(output, dim=1)
        self.top1(pred, data.y)
        return loss


    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None):
        self.maximize=False
        while self.curr_step < self.opt.train_iters:
            if self.curr_step < self.opt.msteps:
                self.maximize=True
                time_start = time.process_time()
                self.train_one_epoch_gc(loader=forget_loader)
                self.save_files['train_time_taken'] += time.process_time() - time_start
                self.eval_gc(loader=test_loader)

            self.maximize=False
            time_start = time.process_time()
            self.train_one_epoch_gc(loader=train_loader)
            self.save_files['train_time_taken'] += time.process_time() - time_start
            self.eval_gc(loader=test_loader)
        return
    
    def unlearn_nc(self, dataset, train_mask, forget_mask):
        self.maximize=False
        while self.curr_step < self.opt.train_iters:
            if self.curr_step < self.opt.msteps:
                self.maximize=True
                time_start = time.process_time()
                self.train_one_epoch(data=dataset, mask=forget_mask)
                self.save_files['train_time_taken'] += time.process_time() - time_start
                self.eval(data=dataset)

            # self.maximize=False
            # time_start = time.process_time()
            # self.train_one_epoch(data=dataset, mask=train_mask)
            # self.save_files['train_time_taken'] += time.process_time() - time_start
            # self.eval(data=dataset)
        return
    

    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.kd_T)+'_'+str(self.opt.alpha)+'_'+str(self.opt.msteps)
        return


class BadT(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.og_model = copy.deepcopy(model)
        self.og_model.to(self.opt.device)
        self.og_model.eval()
        self.random_model = unlearn_func(model, 'EU')
        self.random_model.eval()
        self.kltemp = 1
        opt.train_iters = opt.unlearn_iters
        self.opt.train_iters = opt.unlearn_iters
    

    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)
        
        full_teacher_logits = self.og_model(images)
        unlearn_teacher_logits = self.random_model(images)
        f_teacher_out = torch.nn.functional.softmax(full_teacher_logits / self.kltemp, dim=1)
        u_teacher_out = torch.nn.functional.softmax(unlearn_teacher_logits / self.kltemp, dim=1)
        labels = torch.unsqueeze(infgt, 1)
        overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
        student_out = F.log_softmax(output /self.kltemp, dim=1)
        loss = F.kl_div(student_out, overall_teacher_out)
        
        self.top1(output, target)
        return loss


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        return 


class SSD(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        opt.train_iters = opt.unlearn_iters
        self.opt.train_iters = opt.unlearn_iters


    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None):
        old_iters = self.opt.train_iters
        
        self.opt.train_iters = len(train_loader) + len(forget_loader)
        time_start = time.process_time()
        self.best_model = ssd_tuning(self.model, forget_loader, self.opt.SSDdampening, self.opt.SSDselectwt, train_loader, self.opt.device)
        self.save_files['train_time_taken'] += time.process_time() - time_start
        self.opt.train_iters = old_iters
        return
    
    def unlearn_nc(self, dataset, train_mask, forget_mask): 
        old_iters = self.opt.train_iters
        self.opt.train_iters = len([x for x in train_mask if x]) + len([x for x in forget_mask if x])
        self.best_model = ssd_tuning_nc(self.model, dataset, forget_mask, self.opt.SSDdampening, self.opt.SSDselectwt, train_mask, device=device)
        self.opt.train_iters = old_iters
        return


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.SSDdampening)+'_'+str(self.opt.SSDselectwt)
        return 

def run_ssd(model, dataset, adv_dataset, forget_mask, retain_mask):
    ssd = SSD(opt=opt, model=model)
    start_time = time()
    ssd.unlearn_nc(dataset=dataset, train_mask=retain_mask, forget_mask=forget_mask)
    print(f'SSD unlearning time: {time() - start_time:.2f}s')
    unlearned_model = ssd.best_model

    print('Evaluating SSD unlearned model on the test set')
    test_model(unlearned_model, dataset)
    
    if opt.poison:
        print('Evaluating SSD on the poisoned test set')
        test_model(unlearned_model, adv_dataset)
        compute_success_rate(model=unlearned_model, data=adv_dataset)

def run_scrub(model, dataset, retain_mask, delete_mask):
    scrub = Scrub(opt=opt, model=model)
    start_time = time()
    scrub.unlearn_nc(dataset=dataset, train_mask=retain_mask, forget_mask=delete_mask)
    print(f'SCRUB unlearning time: {time() - start_time:.2f}s')
    unlearned_model = scrub.best_model
    print('Evaluating SCRUB unlearned model on the test set')
    test_model(unlearned_model, dataset)
    
    if opt.poison:
        print('Evaluating SCRUB on the poisoned test set')
        test_model(unlearned_model, adv_test_loader)
        compute_success_rate(model=unlearned_model, loader=adv_test_loader)
