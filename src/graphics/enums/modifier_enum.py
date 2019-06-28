from enum import IntEnum


class ModifierEnum(IntEnum):
    CreatePrimitiveModifier = 0
    FinalizeModelModifier = 1
    TranslateVertexModifier = 2
    TranslateEdgeModifier = 3
    TranslateFaceModifier = 4
    SplitEdgeModifier = 5
    SplitVertexModifier = 6
    ContractVertexPairModifier = 7

