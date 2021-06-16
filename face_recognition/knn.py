#!/usr/bin/env python

import os
import json
import tqdm
import torch
import argparse
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as transforms
import torchvision.utils as tvutils
import torchvision.transforms.functional as tvF

import irse

from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from attgan_nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
from data import check_attribute_conflict

def parse_args():
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    default_num_test = 10
    default_batch_size = 512

    parser.add_argument('-r', '--run', type=str, default='anon', help='results dir')
    parser.add_argument('--test_atts', dest='test_atts', nargs='+', help='test_atts')
    parser.add_argument('--test_ints', dest='test_ints', type=float, nargs='+', help='test_ints')
    parser.add_argument('-b', '--batch_size', type=int, default=default_batch_size, help='batch size')
    parser.add_argument('--num_test', type=int, default=default_num_test, help='number of imgs to test')
    parser.add_argument('--dataroot', type=str, default='./datasets/celeba', help='path to dataset')
    parser.add_argument('-pdb', action='store_true', help='run with pdb')
    return parser.parse_args()
    
    
# Code from AttGAN-PyTorch repo
"""
from this line 
"""
MAX_DIM = 64 * 16  # 1024

class Generator(nn.Module):
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=13, shortcut_layers=1, inject_layers=0, img_size=128):
        super(Generator, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2**enc_layers  # f_size = 4 for 128x128
        
        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn
            )]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)
        
        layers = []
        n_in = n_in + n_attrs  # 1024 + 13
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2**(dec_layers-i-1), MAX_DIM)
                layers += [ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                )]
                n_in = n_out
                n_in = n_in + n_in//2 if self.shortcut_layers > i else n_in
                n_in = n_in + n_attrs if self.inject_layers > i else n_in
            else:
                layers += [ConvTranspose2dBlock(
                    n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh'
                )]
        self.dec_layers = nn.ModuleList(layers)
    
    def encode(self, x):
        z = x
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        return zs
    
    def decode(self, zs, a):
        a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, 1, self.f_size, self.f_size)
        z = torch.cat([zs[-1], a_tile], dim=1)
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            if self.shortcut_layers > i:  # Concat 1024 with 512
                z = torch.cat([z, zs[len(self.dec_layers) - 2 - i]], dim=1)
            if self.inject_layers > i:
                a_tile = a.view(a.size(0), -1, 1, 1) \
                          .repeat(1, 1, self.f_size * 2**(i+1), self.f_size * 2**(i+1))
                z = torch.cat([z, a_tile], dim=1)
        return z
    
    def forward(self, x, a=None, mode='enc-dec'):
        if mode == 'enc-dec':
            assert a is not None, 'No given attribute.'
            return self.decode(self.encode(x), a)
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a)
        raise Exception('Unrecognized mode: ' + mode)

class Discriminators(nn.Module):
    # No instancenorm in fcs in source code, which is different from paper.
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=5, img_size=128):
        super(Discriminators, self).__init__()
        self.f_size = img_size // 2**n_layers
        
        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = nn.Sequential(*layers)
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')
        )
        self.fc_cls = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 13, 'none', 'none')
        )
    
    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_adv(h), self.fc_cls(h)
    
class AttGAN():
    def __init__(self, args):
        self.mode = args.mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.lambda_gp = args.lambda_gp
        
        self.G = Generator(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size
        )
        self.G.train()
        if self.gpu: self.G.cuda()
        # summary(self.G, [(3, args.img_size, args.img_size), (args.n_attrs, 1, 1)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        self.D = Discriminators(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers, args.img_size
        )
        self.D.train()
        if self.gpu: self.D.cuda()
        # summary(self.D, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        if self.multi_gpu:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=args.betas)
        
    def cuda(self):
        self.G.cuda()
    
    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr
    
    def trainG(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = False
        
        zs_a = self.G(img_a, mode='enc')
        img_fake = self.G(zs_a, att_b_, mode='dec')
        img_recon = self.G(zs_a, att_a_, mode='dec')
        d_fake, dc_fake = self.D(img_fake)
        
        if self.mode == 'wgan':
            gf_loss = -d_fake.mean()
        if self.mode == 'lsgan':  # mean_squared_error
            gf_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        gc_loss = F.binary_cross_entropy_with_logits(dc_fake, att_b)
        gr_loss = F.l1_loss(img_recon, img_a)
        g_loss = gf_loss + self.lambda_2 * gc_loss + self.lambda_1 * gr_loss
        
        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()
        
        errG = {
            'g_loss': g_loss.item(), 'gf_loss': gf_loss.item(),
            'gc_loss': gc_loss.item(), 'gr_loss': gr_loss.item()
        }
        return errG
    
    def trainD(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = True
        
        img_fake = self.G(img_a, att_b_).detach()
        d_real, dc_real = self.D(img_a)
        d_fake, dc_fake = self.D(img_fake)
        
        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda() if self.gpu else alpha
                inter = a + alpha * (b - a)
                return inter
            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp
        
        if self.mode == 'wgan':
            wd = d_real.mean() - d_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img_a, img_fake)
        if self.mode == 'lsgan':  # mean_squared_error
            df_loss = F.mse_loss(d_real, torch.ones_like(d_fake)) + \
                      F.mse_loss(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            df_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                      F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        dc_loss = F.binary_cross_entropy_with_logits(dc_real, att_a)
        d_loss = df_loss + self.lambda_gp * df_gp + self.lambda_3 * dc_loss
        
        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()
        
        errD = {
            'd_loss': d_loss.item(), 'df_loss': df_loss.item(), 
            'df_gp': df_gp.item(), 'dc_loss': dc_loss.item()
        }
        return errD
    
    def train(self):
        self.G.train()
        self.D.train()
    
    def eval(self):
        self.G.eval()
        self.D.eval()
    
    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict()
        }
        torch.save(states, path)
    
    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
            print('loaded generator')
        else:
            print('could not load generator')
        if 'D' in states:
            self.D.load_state_dict(states['D'])
            print('loaded discriminator')
        else:
            print('could not load discriminator')
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
            print('loaded optim_G')
        else:
            print('could not load optim_G')
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])
            print('loaded optim_D')
        else:
            print('could not load optim_D')
    
    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)    

class CelebA(data.Dataset):
    def __init__(self, data_path, attr_path, identity_path, transform, selected_attrs):
        super(CelebA, self).__init__()
        assert osp.exists(data_path), data_path + " not found"
        assert osp.exists(attr_path), attr_path + " not found"
        assert osp.exists(identity_path), identity_path + " not found"
        
        self.data_path = data_path

        # these are attributes
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        # in column number form
        # the original attributes of dataset are not what we want
        # the attributes need to be selected
        # attgan_args.attrs are what we need
        # so atts is the indices of useful attributes in annotations
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        # here, we get all image paths
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        # here, we get all the 
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        identity = np.loadtxt(identity_path, usecols=[1], dtype=np.int)
        
        self.images = images
        self.labels = labels
        self.identity = identity
        
        self.transform = transform
                                       
        self.length = len(self.images)
        
    def __getitem__(self, index):

        img = osp.join(self.data_path, self.images[index])
        assert osp.exists(img), img + " not found"
        img = Image.open(img)
        img = self.transform(img)
        att = torch.tensor((self.labels[index] + 1) // 2)
        identity = self.identity[index]

        return img, att, identity
    
    def __len__(self):
        
        return self.length
    
"""
to this line. (code was copied from AttGAN-PyTorch)
"""    


def get_original_image_embeddings(dataloader, feature_extractor, args):
    
    X = list()
    y = list()
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(dataloader)):
            if i >= args.num_test:
                break
            img, attr, identity = batch
            img = img.cuda()
        
            embed = feature_extractor.encode(img)
        
            X.append(embed)
            y.append(identity)
            
        X = torch.cat(X).detach().cpu().numpy()
        y = torch.cat(y).numpy()

    return X, y
    
    
class FeatureExtractor:
    def __init__(self, pretrain_path):
        assert osp.exists(pretrain_path), pretrain_path + " not found"
        self.backbone = irse.IR_50([112, 112])
        self.backbone.load_state_dict(torch.load(pretrain_path))
        self.backbone.eval()
        self.backbone = self.backbone.cuda()

    def encode(self, x):
        with torch.no_grad():
            embed = self.backbone(x)
        return embed
    
    
def resize_img_batch(batch, s1, s2, img_transform):
    out_list = list()
    assert batch.size(-1) == s1
    with torch.no_grad():
        low = -1. 
        high = 1.
        for i in range(len(batch)):
            
            img = batch[i].clamp(min=low, max=high)
            img = img.sub(low).div(max(high - low, 1e-5))
            img = img.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
            img = Image.fromarray(img)
            img = img.resize((s2, s2))            
            img = img_transform(img)
            out_list.append(img)
            
    out_list = torch.stack(out_list)
    
    return out_list
        
    
def evaluate_anonymizer(dataloader, attgan, feature_extractor, img_transform, knn, args):
    
    mismatch = 0
    total = 0
    
    upsample = get_img_transforms(128)
    downsample = get_img_transforms(112)
    
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        if i >= args.num_test:
            break
        img, att_a, identity = batch
        att_a = att_a.cuda()
                
        att_b = att_a.clone()
        
        for a in args.test_atts:
            i = args.attrs.index(a)
            att_b[:, i] = 1 - att_b[:, i]
            att_b = check_attribute_conflict(att_b, args.attrs[i], args.attrs)
            
        with torch.no_grad():
            att_b_ = (att_b * 2 - 1) * args.thres_int
            for a, i in zip(args.test_atts, args.test_ints):
                att_b_[..., args.attrs.index(a)] = att_b_[..., args.attrs.index(a)] * i / args.thres_int
                
            # convert 112 to 128
            img = resize_img_batch(img, 112, 128, upsample)
            tvutils.save_image(img, osp.join(args.run, 'original.jpg'), normalize=True, range=(-1.0, 1.0))
            img = img.cuda()
            anon_img = attgan.G(img, att_b_)
            anon_img = anon_img.cpu()
            
            # convert 128 to 112
            anon_img = resize_img_batch(anon_img, 128, 112, downsample)
            tvutils.save_image(anon_img, osp.join(args.run, 'anon.jpg'), normalize=True, range=(-1.0, 1.0))
            anon_img = anon_img.cuda()

            anon_feature = feature_extractor.encode(anon_img).cpu().numpy()
            
            pred = knn.predict(anon_feature)
            
            identity = identity.numpy()

            mismatch += (pred != identity).astype(np.int).sum()
            total += len(identity)
            
    return mismatch, total
    
def get_img_transforms(image_size):
    
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop([image_size, image_size]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
if __name__ == '__main__':
    
    # get args
    args = parse_args()
    
    if args.pdb:
        import pdb
        pdb.set_trace()

    np.random.seed(0)
    torch.manual_seed(0)

    image_size = 112
    img_transform = get_img_transforms(image_size)

    # attgan args
    with open(osp.join(args.run, '128_shortcut1_inject1_none', 'setting.txt'), 'r') as f:
        attgan_args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
        attgan_args.n_attrs = len(attgan_args.attrs)
        attgan_args.betas = (attgan_args.beta1, attgan_args.beta2)
        attgan_args.test_atts = args.test_atts
        attgan_args.test_ints = args.test_ints
        attgan_args.num_test = args.num_test
        attgan_args.run = args.run
        
    print('attrs')
    print(attgan_args.attrs)
    dataset = CelebA(osp.join(args.dataroot, 'celeba'), 
                    # attribute annotations
                    osp.join(args.dataroot, 'list_attr_celeba.txt'),
                    # identity annotations
                    osp.join(args.dataroot, 'identity_CelebA.txt'),
                    img_transform,
                    attgan_args.attrs)
            
    dataset_size = len(dataset)
    print('dataset size', dataset_size)
    
    dataloader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    feature_extractor = FeatureExtractor(osp.join(args.run, 'pretrain_ir50.pth'))
    
    X, y = get_original_image_embeddings(dataloader, feature_extractor, args)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)

    # create network
    attgan = AttGAN(attgan_args)
    attgan_weights_path = osp.join(args.run, '128_shortcut1_inject1_none', 'checkpoint', 'weights.49.pth')
    assert osp.exists(attgan_weights_path), attgan_weights_path + " not found"
    # load weights
    attgan.cuda()
    attgan.load(attgan_weights_path)
    # eval mode
    attgan.eval()

    dataset = CelebA(osp.join(args.dataroot, 'celeba'), 
                    # attribute annotations
                    osp.join(args.dataroot, 'list_attr_celeba.txt'),
                    # identity annotations
                    osp.join(args.dataroot, 'identity_CelebA.txt'),
                    img_transform,
                    attgan_args.attrs)

    mismatch, total = evaluate_anonymizer(dataloader, attgan, feature_extractor, img_transform, knn, attgan_args)
    print('acc:', mismatch * 100 / total)
    