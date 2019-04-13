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

    @staticmethod
    def _mask_rows(tensor, rows_mask):
        modifier_dim = tensor.shape[1]

        # Smear the masking bits across the entire modifier encoding (to all columns)
        mask = rows_mask.view(-1, 1).repeat(1, modifier_dim)
        tensor = tensor * mask
        return tensor

    def forward(self, imgs, modifiers, non_padded_modifiers_mask):
        batch_dim, ref_imgs_dim = imgs.shape[:2]
        reshaped_imgs = imgs.reshape(-1, *imgs.shape[2:])
        x = self.feature_extractor_block(reshaped_imgs)
        x = x.reshape(batch_dim, ref_imgs_dim, x.shape[1])

        # TODO: Fix comment here
        # src_seq = BATCH x 3 x CNN_out_dim   (3 is for top/front/left, CNN_out_dim=512 or 2048 is ResNet feature vector)..
        # src_pos = [1,2,3] (always, we use all 3 images, no masking..)
        # tgt_seq = BATCH x N x dOUT.   (For each triplet, we have N modifiers, each represented with dOUT dimensions)
        # tgt_pos = [1,2,3] (always, we use all 3 images, no masking..)
        modifiers_cnt = modifiers.shape[1]
        img_mask = torch.arange(1, ref_imgs_dim+1, device=x.device)
        img_mask_per_ref_img = img_mask.repeat(batch_dim, 1)
        modifiers_mask = torch.arange(1, modifiers_cnt+1, device=x.device)
        modifiers_mask = modifiers_mask.repeat(batch_dim, 1)
        src_seq = x
        src_pos = img_mask_per_ref_img
        tgt_pos = modifiers_mask
        tgt_seq = tgt_pos  # self.to_tgt_mask(modifiers_cnt)
        tgt_emb = modifiers

        modifier_encoding = self.img_to_seq_block(src_seq, src_pos, tgt_seq, tgt_pos, tgt_emb)

        modifier_class_id, element_type_tensor, element_pos_tensor, modifier_params = \
            self.modifiers_decoder(modifier_encoding)

        modifier_class_id = self._mask_rows(modifier_class_id, non_padded_modifiers_mask)
        element_type_tensor = self._mask_rows(element_type_tensor, non_padded_modifiers_mask)
        element_pos_tensor = self._mask_rows(element_pos_tensor, non_padded_modifiers_mask)
        modifier_params = self._mask_rows(modifier_params, non_padded_modifiers_mask)

        return modifier_class_id, element_type_tensor, element_pos_tensor, modifier_params
