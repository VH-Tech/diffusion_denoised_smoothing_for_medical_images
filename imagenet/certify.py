import os 
import argparse
import time 
import datetime 
from torchvision import transforms, datasets

from core import Smooth 
from DRM import DiffusionRobustModel
from datasets import get_dataset


from architectures import CLASSIFIERS_ARCHITECTURES, get_architecture
from datasets import get_dataset, DATASETS

from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code

import argparse
import datetime
import numpy as np
import os
import time
import torch
import random

HYPER_DATA_DIR = "/storage/vatsal/datasets/hyper"

def main(args):
    model = DiffusionRobustModel(classifier_name="vit")

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    # dataset = datasets.ImageFolder(root=IMAGENET_DATA_DIR, transform=transform)
    dataset = get_dataset("hyper", 'test', HYPER_DATA_DIR)

    # Get the timestep t corresponding to noise level sigma
    target_sigma = args.sigma * 2
    real_sigma = 0
    t = 0
    while real_sigma < target_sigma:
        t += 1
        a = model.diffusion.sqrt_alphas_cumprod[t]
        b = model.diffusion.sqrt_one_minus_alphas_cumprod[t]
        real_sigma = b / a

    # Define the smoothed classifier 
    smoothed_classifier = Smooth(model, 1000, args.sigma, t)

    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    total_num = 0
    correct = 0
    # for i in range(len(dataset)):
    #     # if i % args.skip != 0:
    #     #     continue

    #     (x, label) = dataset[i]
    #     x = x.cuda()
    #     print("started")
    #     before_time = time.time()
    #     # prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch_size)
  
    #     print(test_loss, test_acc)
    #     after_time = time.time()
    #     print("ended")
    #     correct += int(prediction == label)

    #     time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
    #     total_num += 1

    #     print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
    #         i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
    print("started")
    test_loader = DataLoader(dataset, shuffle=False, batch_size=64,num_workers=4)
    criterion = CrossEntropyLoss().cuda()
    test_loss, test_acc = test(test_loader, model, criterion, args.sigma, t)
    print("Test loss: %.4f, Test accuracy: %.4f" % (test_loss, test_acc))
    # print("sigma %.2f accuracy of smoothed classifier %.4f "%(args.sigma, correct/float(total_num)))


def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float, t: int):
    """
    Function to evaluate the trained model
        :param loader:DataLoader: dataloader (train)
        :param model:torch.nn.Module: the classifer being evaluated
        :param criterion: the loss function
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            if noise_sd == -1:
                #choose randomly a value between 0 and 1
                noise_sd = random.random()

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            # compute output
            outputs = model(inputs, t)
            outputs = outputs.logits
                
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            # top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            
            print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))

        return (losses.avg, top1.avg)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, help="noise hyperparameter")
    parser.add_argument("--skip", type=int, default=10, help="how many examples to skip")
    parser.add_argument("--N0", type=int, default=100, help="number of samples to use")
    parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
    parser.add_argument("--batch_size", type=int, default=1000, help="batch size")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument("--outfile", type=str, help="output file")
    args = parser.parse_args()

    main(args)