from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network, save_trained_config, load_pretrain
from lib.evaluators import make_evaluator
import torch
import torch.multiprocessing
import torch.distributed as dist
import os
from termcolor import colored, cprint
torch.autograd.set_detect_anomaly(True)

if cfg.fix_random:
    torch.manual_seed(0)
    raise NotImplementedError
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def train(cfg, network):
    trainer = make_trainer(cfg, network)
    if not cfg.silent:print("Finish initialize trainer...")

    optimizer = make_optimizer(cfg, network)
    if not cfg.silent:print("Finish initialize optimizer...")
    scheduler = make_lr_scheduler(cfg, optimizer)
    if not cfg.silent:print("Finish initialize lr scheduler...")
    recorder = make_recorder(cfg)
    if not cfg.silent:print("Finish initialize recorder...")
    evaluator = make_evaluator(cfg)
    if not cfg.silent:print("Finish initialize evaluator...")

    begin_epoch = load_model(network,
                             optimizer,
                             scheduler,
                             recorder,
                             cfg.trained_model_dir,
                             resume=cfg.resume)
    if begin_epoch == 0 and cfg.pretrain != '':
        load_pretrain(network, cfg.pretrain)

    set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg,
                                    is_train=True,
                                    is_distributed=cfg.distributed,
                                    max_iter=cfg.ep_iter)
    val_loader = make_data_loader(cfg, is_train=False)

    print(colored(f"[*] Training experiment {cfg.exp_name} started", 'green'))
    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        if cfg.distributed:
            raise NotImplementedError
            # train_loader.batch_sampler.sampler.set_epoch(epoch)

        # train_loader.dataset.epoch = epoch
        recorder.epoch = epoch

        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder,
                       cfg.trained_model_dir, epoch)

        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       last=True)

        if (epoch + 1) % cfg.eval_ep == 0 and cfg.local_rank == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network,
                         cfg.trained_model_dir,
                         resume=cfg.resume,
                         epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def main():
    if cfg.distributed:
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend="nccl",
                                init_method="env://")
        synchronize()

    network = make_network(cfg)
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)
    if cfg.local_rank == 0:
        print('Success!')
        print('='*80)
    # os.system('kill -9 {}'.format(os.getpid()))


if __name__ == "__main__":
    main()
