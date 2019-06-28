import torch
import torch.nn as nn
import torch.nn.functional as F
from app.modifier_visitors.modifier_id_visitor import ModifierIDVisitor
from graphics.enums.element_type_enum import ElementType


class ModifiersDecoder(nn.Module):

    def __init__(self, encoding_size):
        super().__init__()
        modifiers_classes_count = ModifierIDVisitor.max_id() + 1
        element_types_count = len([element_type.value for element_type in ElementType])
        max_selected_element_coords = 9
        max_params = 3

        self.enc_to_class_id = nn.Linear(encoding_size, modifiers_classes_count, bias=True)
        self.enc_to_element_type = nn.Linear(encoding_size, element_types_count, bias=True)
        self.enc_to_selected_element_pos = nn.Sequential(
            nn.Linear(encoding_size, max_selected_element_coords, bias=True),
            nn.Tanh()
        )
        self.enc_to_modifier_params = nn.Sequential(
            nn.Linear(encoding_size, max_params, bias=True),
            nn.Tanh()
        )

        nn.init.xavier_normal_(self.enc_to_class_id.weight)
        nn.init.xavier_normal_(self.enc_to_element_type.weight)
        for module in self.enc_to_selected_element_pos.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
        for module in self.enc_to_modifier_params.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def decode(self, encoding):

        modifier_class_id = self.enc_to_class_id(encoding)
        element_type_tensor = self.enc_to_element_type(encoding)
        element_pos_tensor = self.enc_to_selected_element_pos(encoding)
        modifier_params = self.enc_to_modifier_params(encoding)

        return modifier_class_id, element_type_tensor, element_pos_tensor, modifier_params

    def forward(self, encoding):
        return self.decode(encoding)
