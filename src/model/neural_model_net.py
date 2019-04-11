import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modifiers_decoder import ModifiersDecoder


class NeuralModelNet(nn.Module):

    def __init__(self, feature_extractor, img_to_seq_block, output_dim):

        super().__init__()
        self.feature_extractor_block = feature_extractor
        self.img_to_seq_block = img_to_seq_block
        self.modifiers_decoder = ModifiersDecoder(encoding_size=output_dim)

    def to_tgt_mask(self, n):
        # TODO: Mask calc should be done for a batch
        pad_amount = self.output_dim - n
        mask = F.pad(input=torch.ones(n), pad=(0, pad_amount), value=0)
        mask = mask.unsqueeze(dim=0)
        return mask

    def forward(self, imgs, modifiers):
        batch_dim, ref_imgs_dim = imgs.shape[:2]
        reshaped_imgs = imgs.reshape(-1, *imgs.shape[2:])
        x = self.feature_extractor_block(reshaped_imgs)
        x = x.reshape(batch_dim, ref_imgs_dim, x.shape[1])

        # src_seq = BATCH x 3 x CNN_out_dim   (3 is for top/front/left, CNN_out_dim=512 or 2048 is ResNet feature vector)..
        # src_pos = [1,2,3] (always, we use all 3 images, no masking..)
        # tgt_seq = BATCH x N x dOUT.   (For each triplet, we have N modifiers, each represented with dOUT dimensions)
        # tgt_pos = [1,2,3] (always, we use all 3 images, no masking..)
        modifiers_cnt = modifiers.shape[1]
        img_mask = np.array(list(range(1, ref_imgs_dim+1)))
        img_mask_per_ref_img = np.tile(img_mask, reps=(batch_dim, 1))
        modifiers_mask = np.array(list(range(1, modifiers_cnt + 1)))
        src_seq = x
        src_pos = torch.from_numpy(img_mask_per_ref_img)
        tgt_pos = torch.from_numpy(modifiers_mask).unsqueeze(0)
        tgt_seq = tgt_pos  # self.to_tgt_mask(modifiers_cnt)
        tgt_emb = modifiers

        modifier_encoding = self.img_to_seq_block(src_seq, src_pos, tgt_seq, tgt_pos, tgt_emb)

        modifier_class_id, element_type_tensor, element_pos_tensor, modifier_params = \
            self.modifiers_decoder(modifier_encoding)

        return modifier_class_id, element_type_tensor, element_pos_tensor, modifier_params
