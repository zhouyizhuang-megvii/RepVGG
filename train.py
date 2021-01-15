# MIT License

# Copyright (c) 2020 megvii-model

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import bisect
import multiprocessing
import os
import time

import megengine
import megengine.autodiff as autodiff
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
import megengine.optimizer as optim
import numpy as np

# pylint: disable=import-error
import model as repvgg_model

logging = megengine.logger.get_logger()


def main():
    parser = argparse.ArgumentParser(description="MegEngine ImageNet Training")
    parser.add_argument("-d", "--data", metavar="DIR", help="path to imagenet dataset")
    parser.add_argument(
        "-a",
        "--arch",
        default="RepVGG-A0",
        choices=repvgg_model.func_dict.keys(),
        help="model architecture (default: RepVGG-A0)",
    )
    parser.add_argument(
        "-n",
        "--ngpus",
        default=None,
        type=int,
        help="number of GPUs per node (default: None, use all available GPUs)",
    )
    parser.add_argument(
        "--save",
        metavar="DIR",
        default="output",
        help="path to save checkpoint and log",
    )
    parser.add_argument("--epochs", default=120, help="number of total epochs to run (default: 120)")
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="SIZE",
        default=32,  # 128 (4x batch size is okay on 2080ti)
        type=int,
        help="batch size for single GPU (default: 32)",
    )
    parser.add_argument(
        "--lr",
        metavar="LR",
        default=0.0125,  # 0.05
        type=float,
        help="learning rate for single GPU (default: 0.0125)",
    )
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum (default: 0.9)")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay (default: 0.9)")

    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument(
        "-p",
        "--print-freq",
        default=20,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )

    parser.add_argument("--dist-addr", default="localhost")
    parser.add_argument("--dist-port", default=23456)
    parser.add_argument("--world-size", default=1)
    parser.add_argument("--rank", default=0)

    args = parser.parse_args()

    # create server if is master
    if args.rank <= 0:
        dist.Server(port=args.dist_port)

    # get device count
    with multiprocessing.Pool(1) as pool:
        ngpus_per_node, _ = pool.map(megengine.get_device_count, ["gpu", "cpu"])
    if args.ngpus:
        ngpus_per_node = args.ngpus

    # launch processes
    procs = []
    for local_rank in range(ngpus_per_node):
        p = multiprocessing.Process(
            target=worker,
            kwargs=dict(
                rank=args.rank * ngpus_per_node + local_rank,
                world_size=args.world_size * ngpus_per_node,
                ngpus_per_node=ngpus_per_node,
                args=args,
            ),
        )
        p.start()
        procs.append(p)

    # join processes
    for p in procs:
        p.join()


def worker(rank, world_size, ngpus_per_node, args):
    # pylint: disable=too-many-statements
    if rank == 0:
        os.makedirs(os.path.join(args.save, args.arch), exist_ok=True)
        megengine.logger.set_log_file(os.path.join(args.save, args.arch, "log.txt"))
    # init process group
    if world_size > 1:
        dist.init_process_group(
            master_ip=args.dist_addr,
            port=args.dist_port,
            world_size=world_size,
            rank=rank,
            device=rank % ngpus_per_node,
            backend="nccl",
        )
        logging.info("init process group rank %d / %d", dist.get_rank(), dist.get_world_size())

    if rank:
        [logging.removeHandler(h) for h in logging.handlers]

    # build dataset
    train_dataloader, valid_dataloader = build_dataset(args)
    train_queue = iter(train_dataloader)  # infinite
    steps_per_epoch = 1280000 // (world_size * args.batch_size)

    # build model
    logging.info("Creating model %s", args.arch)
    model = repvgg_model.get_RepVGG_func_by_name(args.arch)(deploy=False)

    # Sync parameters
    if world_size > 1:
        dist.bcast_list_(model.parameters(), dist.WORLD)

    def get_parameters(model):
        params_wd = []
        params_nwd = []
        # no weight decay on bias, bn.gamma (expect for identity branch) and bn.beta
        for n, p in model.named_parameters():
            if n.find("bias") >= 0 or n.find("bn.") >= 0:
                logging.info("NOT include %s %s", n, p.shape)
                params_nwd.append(p)
            else:
                logging.info("    include %s %s", n, p.shape)
                params_wd.append(p)
        return [
            {"params": params_wd},
            {"params": params_nwd, "weight_decay": 0},
        ]

    # Autodiff gradient manager
    gm = autodiff.GradManager().attach(
        model.parameters(),
        callbacks=dist.make_allreduce_cb("MEAN") if world_size > 1 else None,
    )

    # Optimizer
    args.lr = args.lr * world_size  # Linear Scaling Rule
    opt = optim.SGD(
        get_parameters(model),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # train and valid func
    def train_step(image, label):
        with gm:
            logits = model(image)
            loss = F.nn.cross_entropy(logits, label)
            acc1, acc5 = F.topk_accuracy(logits, label, topk=(1, 5))
            gm.backward(loss)
            opt.step().clear_grad()
        return loss, acc1, acc5

    def valid_step(image, label):
        logits = model(image)
        loss = F.nn.cross_entropy(logits, label)
        acc1, acc5 = F.topk_accuracy(logits, label, topk=(1, 5))
        # calculate mean values
        if world_size > 1:
            loss = F.distributed.all_reduce_sum(loss) / world_size
            acc1 = F.distributed.all_reduce_sum(acc1) / world_size
            acc5 = F.distributed.all_reduce_sum(acc5) / world_size
        return loss, acc1, acc5

    # cosine learning rate scheduler
    def adjust_learning_rate(step):
        lr = args.lr * (np.cos(step / (steps_per_epoch * args.epochs) * np.pi) + 1) / 2
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        return lr

    # start training
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    clck = AverageMeter("Time")

    for step in range(0, args.epochs * steps_per_epoch):
        lr = adjust_learning_rate(step)

        t = time.time()

        image, label = next(train_queue)
        image = megengine.tensor(image, dtype="float32")
        label = megengine.tensor(label, dtype="int32")

        loss, acc1, acc5 = train_step(image, label)

        objs.update(loss.item())
        top1.update(100 * acc1.item())
        top5.update(100 * acc5.item())
        clck.update(time.time() - t)

        if step % args.print_freq == 0 and dist.get_rank() == 0:
            logging.info(
                "Epoch %d Step %d, LR %.4f, %s %s %s %s",
                step // steps_per_epoch, step, lr,
                objs, top1, top5, clck,
            )
            objs.reset()
            top1.reset()
            top5.reset()
            clck.reset()

        if (step + 1) % (1 * steps_per_epoch) == 0:  # eval and save frequency
            model.eval()
            _, valid_acc1, valid_acc5 = valid(valid_step, valid_dataloader, args)
            model.train()
            logging.info(
                "Epoch %d Test Acc@1 %.3f, Acc@5 %.3f",
                (step + 1) // steps_per_epoch,
                valid_acc1,
                valid_acc5,
            )
            megengine.save(
                {
                    "epoch": (step + 1) // steps_per_epoch,
                    "state_dict": model.state_dict(),
                },
                os.path.join(args.save, args.arch, "checkpoint.pkl"),
            ) if rank == 0 else None


def valid(func, data_queue, args):
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    clck = AverageMeter("Time")

    t = time.time()
    for step, (image, label) in enumerate(data_queue):
        image = megengine.tensor(image, dtype="float32")
        label = megengine.tensor(label, dtype="int32")

        n = image.shape[0]

        loss, acc1, acc5 = func(image, label)

        objs.update(loss.item(), n)
        top1.update(100 * acc1.item(), n)
        top5.update(100 * acc5.item(), n)
        clck.update(time.time() - t, n)
        t = time.time()

        if step % args.print_freq == 0 and dist.get_rank() == 0:
            logging.info("Test step %d, %s %s %s %s", step, objs, top1, top5, clck)

    return objs.avg, top1.avg, top5.avg


def build_dataset(args):
    train_dataset = data.dataset.ImageNet(args.data, train=True)
    train_sampler = data.Infinite(
        data.RandomSampler(train_dataset, batch_size=args.batch_size, drop_last=True)
    )
    train_dataloader = data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=T.Compose([
                # Baseline Augmentation for small models
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                # T.Normalize(
                #     mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
                # ),  # BGR
                T.ToMode("CHW"),
            ]),
        num_workers=args.workers,
    )
    valid_dataset = data.dataset.ImageNet(args.data, train=False)
    valid_sampler = data.SequentialSampler(
        valid_dataset, batch_size=100, drop_last=False
    )
    valid_dataloader = data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        transform=T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                # T.Normalize(
                #     mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
                # ),  # BGR
                T.ToMode("CHW"),
            ]
        ),
        num_workers=args.workers,
    )
    return train_dataloader, valid_dataloader


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":.3f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    main()