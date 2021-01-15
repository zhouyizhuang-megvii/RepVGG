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
import multiprocessing
import time

# pylint: disable=import-error
import model as repvgg_model

import megengine
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F

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
        "-m", "--model", metavar="PKL", default=None, help="path to model checkpoint"
    )

    parser.add_argument("-j", "--workers", default=2, type=int)
    parser.add_argument(
        "-p",
        "--print-freq",
        default=20,
        type=int,
        metavar="N",
        help="print frequency (default: 20)",
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
        logging.info(
            "init process group rank %d / %d", dist.get_rank(), dist.get_world_size()
        )

    # build model
    model = repvgg_model.get_RepVGG_func_by_name(args.arch)(deploy=False)

    # load pretrained weights
    if args.model is not None:
        logging.info("load from checkpoint %s", args.model)
        checkpoint = megengine.load(args.model)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)

    # convert and save converted weights to file
    megengine.save(model.state_dict(), f"{args.arch}-train.pkl")
    model = repvgg_model.repvgg_convert(model, repvgg_model.get_RepVGG_func_by_name(args.arch))
    megengine.save(model.state_dict(), f"{args.arch}-deploy.pkl")

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

    model.eval()

    # build dataset
    _, valid_dataloader = build_dataset(args)

    _, valid_acc1, valid_acc5 = valid(valid_step, valid_dataloader, args)
    logging.info(
        "Test Acc@1 %.3f, Acc@5 %.3f", valid_acc1, valid_acc5,
    )


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
    train_dataloader = None
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