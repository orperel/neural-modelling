import torch
import torch.nn.functional as F
from app.modifier_visitors.modifier_id_visitor import ModifierIDVisitor, ModifierEnum
from app.modifier_visitors.selected_element_type_visitor import ElementType


class ModifierLabelsDecoder:

    def __init__(self):
        self.modifiers_classes_count = ModifierIDVisitor.max_id() + 1
        self.element_types_count = len([element_type.value for element_type in ElementType])

    @staticmethod
    def _apply_mapping(vec, mapping_table):
        zeros = torch.zeros_like(vec)
        output = torch.zeros_like(vec)
        for modifier_enum, amount in mapping_table.items():
            # Mapped value smeared over all dimensions
            mapped_value = torch.ones_like(vec) * amount
            # Choose mapped value if it matches the modifier index
            output += torch.where(vec == modifier_enum.value, mapped_value, zeros)

        return output

    def get_selected_elements_amount(self, modifier_class_id):
        # TODO: Or - revise to make this block more modifier generalized
        supported_modifiers_mapping = {
            ModifierEnum.TranslateVertexModifier: 1 * 3,
            ModifierEnum.TranslateEdgeModifier: 2 * 3,
            ModifierEnum.TranslateFaceModifier: 3 * 3,
            ModifierEnum.SplitEdgeModifier: 2 * 3,
        }

        output = self._apply_mapping(vec=modifier_class_id, mapping_table=supported_modifiers_mapping)
        return output

    def get_modifier_params_amount(self, modifier_class_id):
        # TODO: Or - revise to make this block more modifier generalized
        supported_modifiers_mapping = {
            ModifierEnum.TranslateVertexModifier: 3,
            ModifierEnum.TranslateEdgeModifier: 3,
            ModifierEnum.TranslateFaceModifier: 3,
            ModifierEnum.SplitEdgeModifier: 0
        }

        output = self._apply_mapping(vec=modifier_class_id, mapping_table=supported_modifiers_mapping)
        return output

    def _extract_and_mask_tensor(self, multivec, start_pos, end_pos):
        """
        Extracts the dimensions from start_pos to end_pos in each row of multivec.
        :param multivec: A tensor of dimensions BATCH x DIMS
        :param start_pos: Tensor of dimensions: DIMS, representing the starting position of each row
        :param end_pos: Tensor of dimensions: DIMS, representing the end position of each row
        :return: The extracted value at the given range, for each row, and a binary mask tensor of the actual values
        extracted (to disregard padding).
        """
        # First get the leftmost and rightmost indices, and extract the relevant columns from multivec
        left_lim = torch.min(start_pos).item()
        right_lim = torch.max(end_pos).item()
        extracted_segment = multivec[:, left_lim:right_lim]

        # modifiers_count represents the batch size, mask_length is the TOTAL amount of relevant columns, but
        # note it may be sparse (we'll make this vector more dense later on).
        # i.e: extracted_segment may contain:
        # [ [1, 2, 3, 0, 0, 0], [0, 0, 0, 3, 4, 5] ].
        # so mask_length is 6, but ultimately we'll want to return: [ [1, 2, 3], [3, 4, 5] ]
        modifiers_count = end_pos.shape[0]
        mask_length = right_lim - left_lim

        # Create a mask of: [start_pos, start_pos + 1, .., end_pos] for EACH row.
        tiled_indices = torch.arange(right_lim - left_lim).repeat(modifiers_count, 1) + left_lim

        # Create a mask of [start_pos[i], start_pos[i], .. start_pos[i], for each row. Same goes for end_pos.
        # The binary mask is obtained where the indices are between the start and end positions.
        left_mask_lim = start_pos.unsqueeze(1).repeat(1, mask_length)
        right_mask_lim = end_pos.unsqueeze(1).repeat(1, mask_length)
        mask = torch.mul(tiled_indices >= left_mask_lim, tiled_indices < right_mask_lim)

        # -- At this point extracted_segment may contain the result, and mask the actual mask --
        # However, extracted_segment may be sparse, so we wish to "dilate" it.
        #                               ---

        # First extract the real length of the dense vector. This is determined by the maximal range: end-start.
        # Then create a mask of dims_per_row: for each row we keep how many "real" columns should be extracted
        # Then we obtain a short binary mask, containing a 1 bit for each real value we should extract.
        # So for input: [ [1, 2, 3, 0, 0, 0], [0, 0, 0, 3, 4, 5] ]
        # We get: real_length = 3, and dims_per_row = [ [ 3, 3, 3], [ 3, 3, 3,]],
        # and the short_mask is [ [1, 1, 1], [1, 1, 1] ]
        real_length = torch.max(end_pos - start_pos).item()
        dims_per_row = (right_mask_lim - left_mask_lim)[:, :real_length]
        short_mask = torch.arange(real_length).repeat(modifiers_count, 1) < dims_per_row

        # We pad the original mask with an inverted mask, and the original extracted_segment with empty zero
        # values, to get "empty dimensions" we can pick from.
        # Later when we use mask_selection, we'll pick those empty values if this specific row has less columns than
        # other rows.
        # (in other words - for modifiers with less parameters, we append a "dummy" mask of ones to make sure all rows
        # contain an equal number of selected columns, but those are empty zero entries)
        inverted_mask = torch.ones_like(short_mask) - short_mask
        padded_mask = torch.cat((mask, inverted_mask), dim=1)
        empty_entries_padding = torch.zeros(modifiers_count, real_length)
        padded_segment = torch.cat((extracted_segment, empty_entries_padding), dim=1)

        # Finally use mask select to extract the real values and obtain a shorter extracted segment.
        # The real dimensions of this tensor are exactly like the padding we added in the previous stage
        # (think of an empty modifier: it would need the entire padding we've added).
        short_extracted_segment = torch.masked_select(padded_segment, padded_mask).reshape_as(empty_entries_padding)

        # This shouldn't be required, but to be on the safe side: zero out elements that fall out of the mask
        short_mask = short_mask.float()
        short_extracted_segment.mul_(short_mask)

        return short_extracted_segment, short_mask

    def decode(self, encoding):
        encoding = encoding.view(-1, encoding.shape[-1])
        m_id_start, m_id_end = 0, self.modifiers_classes_count
        modifier_class_id = torch.argmax(encoding[:, m_id_start:m_id_end], dim=1)

        e_type_start = m_id_end
        e_type_end = m_id_end + self.element_types_count
        e_pos_start = torch.LongTensor([e_type_end])
        e_pos_end = e_type_end + self.get_selected_elements_amount(modifier_class_id)
        m_param_start = e_pos_end
        m_param_end = e_pos_end + self.get_modifier_params_amount(modifier_class_id)

        element_type_tensor = torch.argmax(encoding[:, e_type_start:e_type_end], dim=1)

        element_pos_tensor, element_pos_tensor_mask = \
            self._extract_and_mask_tensor(multivec=encoding, start_pos=e_pos_start, end_pos=e_pos_end)

        modifier_params, modifier_params_mask = \
            self._extract_and_mask_tensor(multivec=encoding, start_pos=m_param_start, end_pos=m_param_end)

        return modifier_class_id, \
               element_type_tensor, \
               element_pos_tensor, \
               element_pos_tensor_mask, \
               modifier_params, \
               modifier_params_mask
