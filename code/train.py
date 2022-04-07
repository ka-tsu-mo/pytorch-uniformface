import argparse
import os
import random
from tqdm import tqdm
import yaml
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import ResNetFace
from losses import ASoftmax, AdditiveAngularMargin, LargeMarginCosine, move_class_centers, uniform_loss
from dataset import FaceImageDataset, collate_fn


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)

def config2str(config):
    experiment = ''
    for k, v in config.items():
        if k not in ['dataset', 'device']:
            if k == 'loss':
                for k_, v_ in v.items():
                    experiment += f'{k_}{v_}_'
            else:
                experiment += f'{k}{v}_'
    return experiment[:-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
            help='path to config yaml file')
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--alpha', type=float)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.alpha is not None:
        config['alpha'] = args.alpha

    seed_everything(config['seed'])

    dataset = FaceImageDataset(config['dataset']['train'])
    loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
            worker_init_fn=seed_worker,
            collate_fn=collate_fn
            )

    device = torch.device(config['device'])
    emb_dim = config['emb_dim']
    model = ResNetFace(config['model_name'], emb_dim).to(device)

    loss_conf = config['loss']
    if loss_conf['name'] == 'arcface':
        s, m, e = loss_conf['scale'], loss_conf['margin'], loss_conf['easy_margin']
        last_fc = AdditiveAngularMargin(dataset.num_classes, emb_dim, s, m, e).to(device)
    elif config['loss']['name'] == 'cosface':
        s, m = loss_conf['scale'], loss_conf['margin']
        last_fc = LargeMarginCosine(dataset.num_classes, emb_dim, s, m).to(device)
    else:
        last_fc = ASoftmax(dataset.num_classes, emb_dim, loss_conf['margin']).to(device)

    lr, momentum, weight_decay = config['lr'], config['momentum'], config['weight_decay']
    no_decay, decay = [], []
    for n, m in model.named_parameters():
        if 'bias' in n or 'prelu' in n:
            no_decay.append(m)
        else:
            decay.append(m)
    if config['decay_w']:
        decay.append(last_fc.class_centers)
    else:
        no_decay.append(last_fc.class_centers)
    params = [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay}]
    optimizer = optim.SGD(params, lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config['milestones'], gamma=0.1)

    exp_name = config2str(config)
    writer = SummaryWriter(os.path.join(args.logdir, exp_name))

    num_iters = 0
    model.train()
    progress_bar = tqdm(total=config['max_iters'])
    torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(5)
    while num_iters < config['max_iters']:
        all_losses = ce_losses = uni_losses = accuracy = count = 0.
        for batch in loader:
            inputs = torch.tensor(batch['images'], device=device)
            labels = torch.tensor(batch['labels'], device=device)
            emb = model(inputs)
            if config['beta'] != 0.:
                move_class_centers(emb, labels, last_fc.class_centers, config['beta'])
            output = last_fc(emb, labels)
            ce_loss = F.cross_entropy(output, labels)
            uni_loss, min_avg, dist = uniform_loss(last_fc.class_centers, labels, config['compute_in_batch'])
            loss = ce_loss + config['alpha']*uni_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # ================ log ================
            with torch.no_grad():
                predict = torch.argmax(output, dim=1)
                correct = torch.eq(predict, labels)
                acc = (100.0 * torch.sum(correct)) / correct.size(0)
            all_losses += loss.item()
            ce_losses += ce_loss.item()
            uni_losses += uni_loss.item()
            accuracy += acc.item()
            if num_iters % 100 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step=num_iters)
                writer.add_scalar('train/accuracy', acc.item(), global_step=num_iters)
                writer.add_scalar('train/ce_loss', ce_loss.item(), global_step=num_iters)
                writer.add_scalar('train/uni_loss', uni_loss.item(), global_step=num_iters)
            # =====================================

            progress_bar.set_postfix({'ce_loss': ce_loss.item(), 'uni_loss': uni_loss.item(), 'dist': dist.item(), 'min_avg': min_avg.item(), 'accuracy': acc.item()})
            progress_bar.update(1)
            num_iters += 1
            count += 1.
            if num_iters == config['max_iters']:
                break

    model_path = os.path.join(args.logdir, exp_name, f'{config["model_name"]}_{config["emb_dim"]}.pt')
    torch.save(model.state_dict(), model_path)
    class_centers_path = os.path.join(args.logdir, exp_name, 'class_centers.pt')
    torch.save(last_fc.state_dict(), class_centers_path)
