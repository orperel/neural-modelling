import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelSeqNet(nn.Module):

    def __init__(self, feature_extractor, img_to_seq_block, output_dim):

        super().__init__()
        self.feature_extractor_block = feature_extractor
        self.img_to_seq_block = img_to_seq_block
        self.output_dim = output_dim

    def to_tgt_mask(self, n):
        pad_amount = self.output_dim - n
        mask = F.pad(input=torch.ones(n), pad=(0, pad_amount), value=0)
        return mask

    def forward(self, top, front, left, modifiers_cnt):
        x1 = self.feature_extractor_block(top)
        x2 = self.feature_extractor_block(front)
        x3 = self.feature_extractor_block(left)

        # src_seq = BATCH x 3 x 512   (3 is for top/front/left, 512 is ResNet feature vector)..
        # src_pos = [1,2,3] (always, we use all 3 images, no masking..)
        # tgt_seq = BATCH x N x dOUT.   (For each triplet, we have N modifiers, each represented with dOUT dimensions)
        # tgt_pos = [1,2,3] (always, we use all 3 images, no masking..)
        img_mask = np.array([1, 2, 3])
        src_seq = torch.cat((x1, x2, x3))
        tgt_seq = self.to_tgt_mask(modifiers_cnt)
        src_pos = torch.from_numpy(img_mask)
        tgt_pos = torch.from_numpy(img_mask)

        y = self.img_to_seq_block(src_seq, src_pos, tgt_seq, tgt_pos)

        return y
