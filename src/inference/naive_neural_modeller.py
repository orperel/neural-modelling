import torch
import torch.nn.functional as F
from inference.modifiers_factory import ModifiersFactory


class NaiveNeuralModeller:

    def __init__(self, model):
        self.model = model
        self.modifiers_factory = ModifiersFactory()

    def process(self, imgs, limit):

        img_features = self.model.encode_img_features(imgs)
        src_seq, src_pos = self.model.prepare_img2seq_enc_input(imgs=imgs, x=img_features)

        tgt_emb = torch.FloatTensor([-0.1, 0.2, 0.1, -0.2, 0.2, -0.1, -0.2, 0.1])
        max_len = 512  # TODO: Or - alter by config

        # Pad all encoding with zeros to ensure they're of size max_len
        tgt_emb = F.pad(tgt_emb, (0, max_len - len(tgt_emb)))
        tgt_emb = tgt_emb.unsqueeze(dim=0).unsqueeze(dim=0)

        modifiers = []

        # TODO: For loop
        for i in range(limit):
            modifiers_cnt = i + 1
            modifiers_mask = torch.arange(1, modifiers_cnt + 1)
            modifiers_mask = modifiers_mask.repeat(1, 1)
            tgt_seq = modifiers_mask
            tgt_pos = modifiers_mask
            next_decoded = self.model.img_to_seq_block.decoder(tgt_seq, tgt_pos, tgt_emb, src_seq, img_features)
            next_decoded = next_decoded[0]
            tgt_emb = torch.cat((tgt_emb, next_decoded), dim=1)

            modifier = self.modifiers_factory.from_tensor(encoding=next_decoded)
            modifier.append(modifier)

        return modifiers

    def __call__(self, imgs):
        return self.process()
