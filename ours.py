"""
UGCLHA: Uncertainty-Guided Consistency Learning for Hard Samples and Ambiguous Regions
Simplified training code for anonymous review.
Full implementation will be released upon acceptance.
"""

import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

# Anonymous project imports (user should provide these modules)
from ugclha.code.dataloaders.datasetsec import (BaseDataSets, TwoStreamBatchSampler, WeakStrongAugment)
from ugclha.code.networks.net_factory import BCP_net
from ugclha.code.utils import ramps, losses
from ugclha.code.val_2D import test_single_volume
from ugclha.utils.loss_all import PAL
from ugclha.utils.zhifangtu import enhance_min_to_max
from ugclha.utils.vatloss import VAT2d_v2_MT
from ugclha.utils import feature_memory, contrastive_losses, cross_contra
from ugclha.utils.historybank import WeightedHistoricalMinImageBank

# ==================== Data root (user must set) ====================
DATA_ROOT = "./data"   # Replace with your dataset path

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default=DATA_ROOT, help='dataset root')
parser.add_argument('--exp', type=str, default='ugclha_exp', help='experiment name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000)
parser.add_argument('--max_iterations', type=int, default=30000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.02)
parser.add_argument('--image_size', type=list, default=[256, 256])
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--labeled_bs', type=int, default=8)
parser.add_argument('--labelnum', type=float, default=0.1)
parser.add_argument('--u_weight', type=float, default=0.5)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--consistency', type=float, default=1)
parser.add_argument('--consistency_rampup', type=float, default=150.0)
parser.add_argument('--magnitude', type=float, default='10.0')
parser.add_argument('--s_param', type=int, default=6)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--h_size', type=int, default=4)
parser.add_argument('--w_size', type=int, default=4)
parser.add_argument('--top_num', type=int, default=4)
parser.add_argument('--dataset', type=str, default='thy')
parser.add_argument('--length', type=int, default='1024')
args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=2)

# ==================== Helper functions ====================
def update_ema_variables_ave(model, model1, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step*10 + 1), alpha)
    for ema_param, param, param1 in zip(ema_model.parameters(), model.parameters(), model1.parameters()):
        ema_param.data.mul_(alpha).add_(((param.data+param1.data)/2)*(1 - alpha))

def update_ema_bn_variables_ave(model, model1, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param, param1 in zip(ema_model.buffers(), model.buffers(), model1.buffers()):
        ema_param.data = ema_param.data * alpha + ((param.data+param1.data)/2)*(1 - alpha)

def generate_top_k_threshold(con, proportion, choice):
    con_flat = con.view(-1)
    k = max(1, int(con_flat.numel() * proportion))
    if choice == 0:
        topk_values, _ = torch.topk(con_flat, k)
        threshold = topk_values.min()
    else:
        topk_values, _ = torch.topk(con_flat, k, largest=False)
        threshold = topk_values.max()
    return threshold

def D(p, z, version='simplified'):
    if version == 'original':
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()
    else:
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

def colorful_spectrum_mix_batch(img1_batch, img2_batch, alpha):
    n, c, h, w = img1_batch.shape
    h_crop = int(0.2 * h)
    w_crop = int(0.2 * w)
    h_start = (h // 2) - (h_crop // 2)
    w_start = (w // 2) - (w_crop // 2)
    h_end = h_start + h_crop
    w_end = w_start + w_crop

    img1_fft = torch.fft.fftn(img1_batch, dim=(-2, -1))
    img2_fft = torch.fft.fftn(img2_batch, dim=(-2, -1))
    img1_abs = torch.abs(img1_fft)
    img1_pha = torch.angle(img1_fft)
    img2_abs = torch.abs(img2_fft)
    img2_pha = torch.angle(img2_fft)

    img1_abs = torch.fft.fftshift(img1_abs, dim=(-2, -1))
    img2_abs = torch.fft.fftshift(img2_abs, dim=(-2, -1))

    mask = torch.zeros((h, w), dtype=torch.bool, device=img1_batch.device)
    mask[h_start:h_end, w_start:w_end] = 1
    mask_batch = mask.view(1, 1, h, w).expand(n, c, h, w)

    img1_abs_mixed = img1_abs.clone()
    img2_abs_mixed = img2_abs.clone()
    img1_abs_mixed[mask_batch] = alpha * img2_abs[mask_batch] + (1 - alpha) * img1_abs[mask_batch]
    img2_abs_mixed[mask_batch] = alpha * img1_abs[mask_batch] + (1 - alpha) * img2_abs[mask_batch]

    img1_abs_mixed = torch.fft.ifftshift(img1_abs_mixed, dim=(-2, -1))
    img2_abs_mixed = torch.fft.ifftshift(img2_abs_mixed, dim=(-2, -1))

    img1_mixed = img1_abs_mixed * torch.exp(1j * img1_pha)
    img2_mixed = img2_abs_mixed * torch.exp(1j * img2_pha)
    img1_mixed = torch.real(torch.fft.ifftn(img1_mixed, dim=(-2, -1)))
    img2_mixed = torch.real(torch.fft.ifftn(img2_mixed, dim=(-2, -1)))
    img1_mixed = torch.clamp(img1_mixed, 0, 255)
    img2_mixed = torch.clamp(img2_mixed, 0, 255)
    return img1_mixed, img2_mixed

def to_one_hot(tensor, nClasses):
    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size, device=tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot

def softmax_mse_loss(input_logits, target_logits):
    input_softmax = F.softmax(input_logits, dim=1)
    return (input_softmax - target_logits) ** 2

def mix_mse_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False, diff_mask=None):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    img_l_onehot = to_one_hot(img_l.unsqueeze(1), 2)
    patch_l_onehot = to_one_hot(patch_l.unsqueeze(1), 2)
    mse_loss = torch.mean(softmax_mse_loss(net3_output, img_l_onehot), dim=1) * mask * image_weight
    mse_loss += torch.mean(softmax_mse_loss(net3_output, patch_l_onehot), dim=1) * patch_mask * patch_weight
    loss = torch.sum(diff_mask * mse_loss) / (torch.sum(diff_mask) + 1e-16)
    return loss

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {'net': net.state_dict(), 'opt': optimizer.state_dict()}
    torch.save(state, str(path))

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    # For simplicity, largest component post-processing is omitted here.
    # In full code, get_ACDC_2DLargestCC is called when nms=1.
    return probs

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    return loss_dice, loss_ce

# ==================== Pre-training ====================
def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model = BCP_net(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,
                            transform=transforms.Compose([WeakStrongAugment(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_len = len(db_train)
    labeled_len = int(total_len * args.labelnum)
    unlabeled_len = total_len - labeled_len
    labeled_train_set, unlabeled_train_set = random_split(db_train, [labeled_len, unlabeled_len],
                                                          generator=torch.Generator().manual_seed(0))
    labeled_idxs = labeled_train_set.indices
    unlabeled_idxs = unlabeled_train_set.indices

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre-training")

    model.train()
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for sampled_batch in trainloader:
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)

            net_input = img_a * img_mask + img_b * (1 - img_mask)
            net_input2 = img_b * img_mask + img_a * (1 - img_mask)
            out_mixl, _ = model(net_input)
            out_mixl2, _ = model(net_input2)

            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            loss_dice1, loss_ce1 = mix_loss(out_mixl2, lab_b, lab_a, loss_mask, u_weight=1.0, unlab=True)
            loss = (loss_dice + loss_ce + loss_dice1 + loss_ce1) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/total_loss', loss, iter_num)

            if iter_num % 200 == 0:
                # Validation and logging (simplified)
                model.eval()
                metric_list = 0.0
                for _, sampled_batch_val in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch_val["image"].unsqueeze(1),
                                                  sampled_batch_val["label"].unsqueeze(1), model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                if performance > best_performance:
                    best_performance = performance
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_best_path)
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break
    writer.close()

# ==================== Self-training (Core: three modules) ====================
def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model_1 = BCP_net(in_chns=1, class_num=num_classes)
    model_2 = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,
                            transform=transforms.Compose([WeakStrongAugment(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_len = len(db_train)
    labeled_len = int(total_len * args.labelnum)
    unlabeled_len = total_len - labeled_len
    labeled_train_set, unlabeled_train_set = random_split(db_train, [labeled_len, unlabeled_len],
                                                          generator=torch.Generator().manual_seed(0))
    labeled_idxs = labeled_train_set.indices
    unlabeled_idxs = unlabeled_train_set.indices

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    load_net(ema_model, pre_trained_model)
    load_net_opt(model_1, optimizer1, pre_trained_model)
    load_net_opt(model_2, optimizer2, pre_trained_model)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self-training")
    adv_loss = VAT2d_v2_MT(epi=args.magnitude)   # Virtual adversarial perturbation

    model_1.train()
    model_2.train()
    ema_model.train()

    # Contrastive learning modules
    prototype_memory = feature_memory.FeatureMemory(elements_per_class=32, n_classes=num_classes)
    inconsistent_contrast = cross_contra.InconsistentRegionContrast(temperature=0.07, weight=0.5)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    historical_bank = WeightedHistoricalMinImageBank(
        max_size=10, num_models=2, selection_strategy='hybrid',
        entropy_coeff=0.8, similarity_coeff=0.2
    )
    enhancement_config = {
        'method': 'hist_match_opencv',
        'use_historical_prob': 1.0,
        'enable_bidirectional': True,
        'min_history_for_use': 3,
    }

    for _ in iterator:
        for sampled_batch in trainloader:
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            volume_batch_strong, _ = sampled_batch['image_strong'].cuda(), sampled_batch['label_strong'].cuda()
            volume_batch_strong_2, _ = sampled_batch['image_strong_2'].cuda(), sampled_batch['label_strong_2'].cuda()

            img_a = volume_batch[:labeled_sub_bs]
            img_b = volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a = volume_batch[args.labeled_bs:args.labeled_bs+unlabeled_sub_bs]
            uimg_b = volume_batch[args.labeled_bs+unlabeled_sub_bs:]
            lab_a = label_batch[:labeled_sub_bs]
            lab_b = label_batch[labeled_sub_bs:args.labeled_bs]

            img_a_s = volume_batch_strong[:labeled_sub_bs]
            img_b_s = volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s = volume_batch_strong[args.labeled_bs:args.labeled_bs+unlabeled_sub_bs]
            uimg_b_s = volume_batch_strong[args.labeled_bs+unlabeled_sub_bs:]

            img_a_s2 = volume_batch_strong_2[:labeled_sub_bs]
            img_b_s2 = volume_batch_strong_2[labeled_sub_bs:args.labeled_bs]
            uimg_a_s2 = volume_batch_strong_2[args.labeled_bs:args.labeled_bs+unlabeled_sub_bs]
            uimg_b_s2 = volume_batch_strong_2[args.labeled_bs+unlabeled_sub_bs:]

            with torch.no_grad():
                pre_a, _ = ema_model(uimg_a)
                pre_b, _ = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)

            img_mask, loss_mask = generate_mask(img_a)

            # Build mixed inputs (same as original)
            net_input_unl_1w = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l_1w = img_b * img_mask + uimg_b * (1 - img_mask)
            net_input_w = torch.cat([net_input_unl_1w, net_input_l_1w], dim=0)

            net_input_unl_1 = uimg_a_s * img_mask + img_a_s * (1 - img_mask)
            net_input_l_1 = img_b_s * img_mask + uimg_b_s * (1 - img_mask)
            net_input_1 = torch.cat([net_input_unl_1, net_input_l_1], dim=0)

            net_input_unl_2 = uimg_a_s2 * img_mask + img_a_s2 * (1 - img_mask)
            net_input_l_2 = img_b_s2 * img_mask + uimg_b_s2 * (1 - img_mask)
            net_input_2 = torch.cat([net_input_unl_2, net_input_l_2], dim=0)

            # ========== Module 1: Multi-source uncertainty (model discrepancy + augmentation) ==========
            out_unl_1w, _ = model_1(net_input_unl_1w)
            out_l_1w, _ = model_1(net_input_l_1w)
            out_1w = torch.cat([out_unl_1w, out_l_1w], dim=0)
            out_soft_1w = torch.softmax(out_1w, dim=1)
            out_pseudo_1w = torch.argmax(out_soft_1w.detach(), dim=1, keepdim=False)

            out_unl_2w, _ = model_2(net_input_unl_1w)
            out_l_2w, _ = model_2(net_input_l_1w)
            out_2w = torch.cat([out_unl_2w, out_l_2w], dim=0)
            out_soft_2w = torch.softmax(out_2w, dim=1)
            out_pseudo_2w = torch.argmax(out_soft_2w.detach(), dim=1, keepdim=False)

            # Compute entropy and select hard samples (max/min entropy)
            entropy_1w = -torch.sum(out_soft_1w.detach() * torch.log2(out_soft_1w.detach() + 1e-10), dim=1)
            w1_wmean_batch = torch.mean(entropy_1w, dim=(1, 2))
            min_idx_1 = torch.argmin(w1_wmean_batch)
            max_idx_1 = torch.argmax(w1_wmean_batch)

            entropy_2w = -torch.sum(out_soft_2w.detach() * torch.log2(out_soft_2w.detach() + 1e-10), dim=1)
            w2_wmean_batch = torch.mean(entropy_2w, dim=(1, 2))
            min_idx_2 = torch.argmin(w2_wmean_batch)
            max_idx_2 = torch.argmax(w2_wmean_batch)

            min_img1 = net_input_w[min_idx_1].unsqueeze(0)
            min_img2 = net_input_w[min_idx_2].unsqueeze(0)
            if max_idx_1 != max_idx_2:
                mean_var = (w1_wmean_batch - w2_wmean_batch) ** 2
                max_idx = torch.argmax(mean_var)
            else:
                max_idx = max_idx_1
            max_img1 = net_input_w[max_idx].unsqueeze(0)
            max_img2 = net_input_w[max_idx].unsqueeze(0)

            # Store hard samples into historical bank
            historical_bank.add_min_images(model_idx=0, min_img=min_img1, entropy=w1_wmean_batch[min_idx_1].item(), current_max_img=max_img1)
            historical_bank.add_min_images(model_idx=1, min_img=min_img2, entropy=w2_wmean_batch[min_idx_2].item(), current_max_img=max_img2)

            # Decide to use historical or current
            use_historical = (enhancement_config['use_historical_prob'] > random.random() and
                              len(historical_bank.historical_min_images[0]) >= enhancement_config['min_history_for_use'])
            if use_historical:
                min_img1 = historical_bank.get_weighted_min_image(model_idx=0, current_max_img=max_img1, default_img=min_img1)
                min_img2 = historical_bank.get_weighted_min_image(model_idx=1, current_max_img=max_img2, default_img=min_img2)

            # ========== Module 2: Histogram matching, Fourier amplitude mixing and VAT ==========
            # Histogram matching
            min_img1_enhanced = enhance_min_to_max(min_img1, max_img1, method='hist_match_opencv')
            max_img1_enhanced = enhance_min_to_max(max_img1, min_img1, method='hist_match_opencv')
            min_img2_enhanced = enhance_min_to_max(min_img2, max_img2, method='hist_match_opencv')
            max_img2_enhanced = enhance_min_to_max(max_img2, min_img2, method='hist_match_opencv')

            with torch.no_grad():
                ema_enhance1_pre, _ = ema_model(torch.cat([min_img1, max_img1], dim=0))
                ema_enhance1_plab = get_ACDC_masks(ema_enhance1_pre, nms=1)
                ema_enhance2_pre, _ = ema_model(torch.cat([min_img2, max_img2], dim=0))
                ema_enhance2_plab = get_ACDC_masks(ema_enhance2_pre, nms=1)

            enhan_model1_pre, _ = model_1(torch.cat([min_img1_enhanced, max_img1_enhanced], dim=0))
            enhan_model1_soft = torch.softmax(enhan_model1_pre, dim=1)
            enhan_loss_1 = dice_loss(enhan_model1_soft, ema_enhance1_plab.unsqueeze(1))

            enhan_model2_pre, _ = model_2(torch.cat([min_img2_enhanced, max_img2_enhanced], dim=0))
            enhan_model2_soft = torch.softmax(enhan_model2_pre, dim=1)
            enhan_loss_2 = dice_loss(enhan_model2_soft, ema_enhance2_plab.unsqueeze(1))

            # Fourier amplitude mixing
            a = 0.5
            min_img1_mp, max_img1_mp = colorful_spectrum_mix_batch(min_img1, max_img1, alpha=a)
            min_img2_mp, max_img2_mp = colorful_spectrum_mix_batch(min_img2, max_img2, alpha=a)

            mp_model1_pre, _ = model_1(torch.cat([min_img1_mp, max_img1_mp], dim=0))
            mp_model1_soft = torch.softmax(mp_model1_pre, dim=1)
            mp_loss_1 = dice_loss(mp_model1_soft, ema_enhance1_plab.unsqueeze(1))

            mp_model2_pre, _ = model_2(torch.cat([min_img2_mp, max_img2_mp], dim=0))
            mp_model2_soft = torch.softmax(mp_model2_pre, dim=1)
            mp_loss_2 = dice_loss(mp_model2_soft, ema_enhance2_plab.unsqueeze(1))
            adv_loss = VAT2d_v2_MT(epi=args.magnitude)

            loss_lds1 = adv_loss(model_1, ema_model, mixall[2:])
            loss_lds2 = adv_loss(model_2, ema_model, mixall[2:])

            # ========== Module 3: Dual contrastive learning (prototype & discrepancy) ==========
            # Feature extraction and contrastive losses (simplified, same as original)
            _, feature_u = model_1(torch.cat([min_img1, max_img1], dim=0))   # reuse
            _, feature_k = model_2(torch.cat([min_img2, max_img2], dim=0))

            # Prototype contrast
            contrastive_loss1 = contrastive_losses.contrastive_class_to_class_learned_memory(model_1, pred_feat_unlabeled_1, pseudo_label1, num_classes, prototype_memory.memory)
            contrastive_loss2 = contrastive_losses.contrastive_class_to_class_learned_memory(model_2, pred_feat_unlabeled_2, pseudo_label2, num_classes, prototype_memory.memory)

            # Inconsistency contrast (discrepancy)
            inconsistent_loss1 = inconsistent_contrast.compute_inconsistent_loss(feature_u, feature_k, mask_diff1)
            inconsistent_loss2 = inconsistent_contrast.compute_inconsistent_loss(feature_u, feature_k, mask_diff2)

            # ========== Final loss composition (same as original) ==========
            # ... (the rest of the loss computation is identical to original code, omitted here for brevity)
            # Please refer to original implementation for complete loss aggregation.

            # For readability, we skip the exact loss summation but the idea is:
            total_loss = (loss_1 + loss_2)  # loss_1 and loss_2 include all terms (consistency, mse, PAL, LDS, contrastive, etc.)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            total_loss.backward()
            optimizer1.step()
            optimizer2.step()

            update_ema_variables_ave(model_1, model_2, ema_model, 0.99, iter_num)

            iter_num += 1
            if iter_num % 200 == 0:
                # validation and logging (same as original)
                pass

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break
    writer.close()

if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    pre_snapshot_path = f"./model/{args.exp}_{args.labelnum}_labeled/pre_train"
    self_snapshot_path = f"./model/{args.exp}_{args.labelnum}_labeled/self_train"
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)