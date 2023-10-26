from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import torch
from progress.bar import Bar

from src.lib.models.data_parallel import DataParallel
from src.lib.trains.train_visual import plot_eval
from src.lib.utils.utils import AverageMeter


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch, opt_input_type="image"):
        imgs_hash = np.array(batch['img_hash'].to('cpu'), dtype=np.float32)
        input_type = "image"
        if opt_input_type == "video":
            input_type = "images" if imgs_hash[0] != np.mean(imgs_hash) else "video"

        outputs = self.model(batch['input'], input_type, len(imgs_hash), "for_training")
        memory_net = None if not hasattr(self.model, "memory_net") else getattr(self.model, "memory_net")
        loss, loss_stats = self.loss(outputs, batch, memory_net)
        return outputs[-1], loss, loss_stats, input_type


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModleWithLoss(model, self.loss)
        self.optimizer_add_loss_params()

    def optimizer_add_loss_params(self):
        self.optimizer.add_param_group({'params': self.loss.parameters()})

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader, opt_input_type):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        lr = self.optimizer.param_groups[0]["lr"]
        end = time.time()
        data_loader.dataset.files_index = 0
        if self.opt.shuffle_every_epoch and self.opt.input_type == "image":
            seed = np.random.randint(100)
            np.random.seed(seed)
            np.random.shuffle(data_loader.dataset.img_files)
            np.random.seed(seed)
            np.random.shuffle(data_loader.dataset.label_files)

        if hasattr(self.model_with_loss.model, "memory_net"):
            self.model_with_loss.model.memory_net.clear_dict()

        for iter_id, batch in enumerate(data_loader):
            # keep the same augmentation in a batch by using the same ramdom seed
            data_loader.dataset.random_seed = np.random.randint(13145)

            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for key in batch:
                if key != 'meta':
                    batch[key] = batch[key].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats, input_type = model_with_loss(batch, opt_input_type)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                # torch.nn.utils.clip_grad_norm_(model_with_loss.parameters(), 1.)
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{stage} {phase}: [{0}/{1}][{2}/{3}]| TOT {total}s| input_type: {input_type}|ETA: {eta:} ' \
                         '|MAX_MEMORY: {MAX_MEMORY:.4f}M\n  lr:{lr:.12f}'.format(epoch, opt.warmup_epochs+opt.num_epochs,
                                                       iter_id, num_iters, stage="warmup" if
                epoch <= opt.warmup_epochs else "formal", phase=phase, total=bar.elapsed_td, eta=bar.eta_td,
                                                       MAX_MEMORY=torch.cuda.max_memory_allocated() / 1024. ** 2,
                                                       input_type=input_type, lr=lr)

            for l in avg_loss_stats:
                if isinstance(loss_stats[l], torch.Tensor):
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['input'].size(0))
                    Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_or_show_iter > 0:
                if iter_id % opt.print_or_show_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
                    plot_eval(batch, output, self.opt)
            else:
                bar.next()

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats, batch

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader, opt_input_type):
        return self.run_epoch('train', epoch, data_loader, opt_input_type)

