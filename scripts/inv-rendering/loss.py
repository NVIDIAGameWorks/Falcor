import torch
import math


def compute_render_loss_L2(img, target_img, weight=1.0):
    diff = img - target_img
    diff = diff.nan_to_num(nan=0)
    loss = 0.5 * torch.square(diff).sum()
    return loss * weight


def compute_render_loss_L1(img, target_img, weight=1.0):
    diff = img - target_img
    diff = diff.nan_to_num(nan=0)
    diff = diff.abs()
    loss = diff.sum()
    return loss * weight


def downsample(input):
    if input.size(0) % 2 == 1:
        input = torch.cat((input, torch.unsqueeze(input[-1, :], 0)), dim=0)
    if input.size(1) % 2 == 1:
        input = torch.cat((input, torch.unsqueeze(input[:, -1], 1)), dim=1)
    return (
        input[0::2, 0::2, :]
        + input[1::2, 0::2, :]
        + input[0::2, 1::2, :]
        + input[1::2, 1::2, :]
    ) * 0.25


def build_pyramid(img):
    level = int(min(math.log2(img.shape[0]), math.log2(img.shape[1]))) + 1
    level = min(5, level)
    imgs = []
    for i in range(level):
        imgs.append(img)
        if i < level - 1:
            img = downsample(img)
    return imgs


def compute_render_loss_pyramid_L1(img, target_pyramid, weight=1.0):
    img_pyramid = build_pyramid(img)
    level = len(img_pyramid)
    loss = 0.0
    for i in range(level):
        loss = loss + compute_render_loss_L1(
            img_pyramid[i], target_pyramid[i], weight
        ) * (4.0**i)
    return loss


def compute_render_loss_pyramid_L2(img, target_pyramid, weight=1.0):
    img_pyramid = build_pyramid(img)
    level = len(img_pyramid)
    loss = 0.0
    for i in range(level):
        loss = loss + compute_render_loss_L2(
            img_pyramid[i], target_pyramid[i], weight
        ) * (4.0**i)
    return loss
