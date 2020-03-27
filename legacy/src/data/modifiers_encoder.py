import torch
from src.graphics.enums.element_type_enum import ElementType
from src.app.modifier_visitors.modifier_id_visitor import ModifierIDVisitor
from src.app.modifier_visitors.modifier_params_visitor import ModifierParamsVisitor
from src.app.modifier_visitors import SelectedElementPositionVisitor
from src.app.modifier_visitors.selected_element_type_visitor import SelectedElementTypeVisitor


class ModifiersEncoder:

    def __init__(self):
        self.modifier_id_visitor = ModifierIDVisitor()
        self.selected_element_type_visitor = SelectedElementTypeVisitor()
        self.selected_element_position_visitor = SelectedElementPositionVisitor()
        self.modifier_params_visitor = ModifierParamsVisitor()

    @staticmethod
    def to_one_hot(n, k):
        """
        :param n: Number of classes (i.e: for 0,1,2,3 n = 4)
        :param k: Index of dimension to set to 1
        :return: One hot PyTorch tensor
        """
        idx = torch.tensor([k])
        one_hot = torch.zeros(n).scatter_(0, idx, 1.)
        return one_hot

    def encode_modifier_id(self, modifier):
        modifier_id = self.modifier_id_visitor.visit(modifier)
        modifier_id = self.to_one_hot(n=self.modifier_id_visitor.max_id() + 1, k=modifier_id)
        return modifier_id

    def encode_selected_element_type(self, modifier):
        selected_element_type = self.selected_element_type_visitor.visit(modifier).value
        selected_element_type = self.to_one_hot(n=len(ElementType), k=selected_element_type)
        return selected_element_type

    def encode_selected_element_pos(self, modifier):
        selected_element_pos = self.selected_element_position_visitor.visit(modifier)
        return torch.FloatTensor(selected_element_pos).reshape(-1)

    def encode_modifier_params(self, modifier):
        modifier_params = self.modifier_params_visitor.visit(modifier)
        return torch.FloatTensor(modifier_params).reshape(-1)

    def encode(self, modifier):

        modifier_id_tensor = self.encode_modifier_id(modifier)
        element_type_tensor = self.encode_selected_element_type(modifier)
        element_pos_tensor = self.encode_selected_element_pos(modifier)
        modifier_params = self.encode_modifier_params(modifier)

        return torch.cat(tensors=(modifier_id_tensor, element_type_tensor, element_pos_tensor, modifier_params),
                         dim=0)
