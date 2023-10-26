from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib.resources as res
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import _init_paths
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from src.lib.trains.model_param import set_parameter
from src.lib import cfg
from src.lib.datasets.dataset_factory import get_dataset
from src.lib.logger import Logger
from src.lib.models.model import create_model, load_model, save_model
from src.lib.opts import opts
from src.lib.trains.train_factory import train_factory


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)
    f = res.open_text(cfg, os.path.basename(opt.data_cfg))
    # f = open(f)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)

    opt = opts().update_dataset_info_and_set_heads(opt, dataset)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch_scale, opt.arch, opt.heads, opt.head_conv, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    if opt.warmup_epochs > 0:
        warm_up_params = [torch.tensor([1.], requires_grad=True)]
        if len(opt.warmup_keys) > 0:
            warm_up_params = set_parameter(opt.warmup_keys, model, opt.warmup_lr, 0.)
        warmup_optimizer = torch.optim.Adam(warm_up_params, opt.warmup_lr)

    start_epoch = 0
    # Get dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer if opt.warmup_epochs == 0 else warmup_optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.pretrained:
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1 + opt.warmup_epochs):
        if epoch in opt.input_type_switch_frq:
            if opt.input_type == 'video':
                print("opt input_type change: ", opt.input_type, '=>', "image")
                opt.input_type = "image"
            else:
                print("opt input_type change: ", opt.input_type, '=>', "video")
                opt.input_type = 'video'

        # print(opt.input_type)
        if 0 < opt.warmup_epochs == epoch - 1:
            trainer.optimizer = torch.optim.Adam(model.parameters(), opt.lr)
            trainer.optimizer_add_loss_params()

        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader, opt.input_type)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, trainer.optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, trainer.optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, trainer.optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, trainer.optimizer)

    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
