import torch
import torch.nn as nn
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

    def encode_img_features(self, imgs):
        batch_dim, ref_imgs_dim = imgs.shape[:2]
        reshaped_imgs = imgs.reshape(-1, *imgs.shape[2:])
        x = self.feature_extractor_block(reshaped_imgs)
        x = x.reshape(batch_dim, ref_imgs_dim, x.shape[1])

        return x

    def prepare_img2seq_enc_input(self, imgs, x):
        batch_dim, ref_imgs_dim = imgs.shape[:2]
        img_mask = torch.arange(1, ref_imgs_dim + 1, device=x.device)
        img_mask_per_ref_img = img_mask.repeat(batch_dim, 1)
        src_seq = x
        src_pos = img_mask_per_ref_img  # src_pos = tensor[1,2,3] (always, we use all ref images, no masking..)

        return src_seq, src_pos

    def prepare_img2seq_dec_input(self, imgs, x, modifiers):
        batch_dim, ref_imgs_dim = imgs.shape[:2]
        modifiers_cnt = modifiers.shape[1]
        modifiers_mask = torch.arange(1, modifiers_cnt+1, device=x.device)
        modifiers_mask = modifiers_mask.repeat(batch_dim, 1)
        tgt_pos = modifiers_mask    # tgt_pos = tensor[1,2,3, ..,  #Maximal Num of Modifiers in batch]
        tgt_seq = tgt_pos           # Differs from traditional Transformer, we deal with embeddings, not "words"
        tgt_emb = modifiers         # The actual decodings we feed

        return tgt_pos, tgt_seq, tgt_emb

    def forward(self, imgs, modifiers, non_padded_modifiers_mask):

        # Extract features from images
        x = self.encode_img_features(imgs)

        # Convert input to format understandable by img2seq model (currently only Transformer is supported)
        src_seq, src_pos = self.prepare_img2seq_enc_input(imgs, x)
        tgt_pos, tgt_seq, tgt_emb = self.prepare_img2seq_dec_input(imgs, x, modifiers)

        # Obtain encodings for modifier predictions
        modifier_encoding = self.img_to_seq_block(src_seq, src_pos, tgt_seq, tgt_pos, tgt_emb)

        # Decode actual values from each modifier
        modifier_class_id, element_type_tensor, element_pos_tensor, modifier_params = \
            self.modifiers_decoder(modifier_encoding)

        # Apply masks for stale modifier entries
        modifier_class_id = self._mask_rows(modifier_class_id, non_padded_modifiers_mask)
        element_type_tensor = self._mask_rows(element_type_tensor, non_padded_modifiers_mask)
        element_pos_tensor = self._mask_rows(element_pos_tensor, non_padded_modifiers_mask)
        modifier_params = self._mask_rows(modifier_params, non_padded_modifiers_mask)

        return modifier_class_id, element_type_tensor, element_pos_tensor, modifier_params
