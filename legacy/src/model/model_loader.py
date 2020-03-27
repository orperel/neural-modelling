import torch
import torch.nn as nn
import torchvision
from src.model.transformer.Models import Transformer
from src.model import NeuralModelNet


def _load_feature_block(architecture_name, freeze_layers=False):

    if architecture_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif architecture_name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif architecture_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif architecture_name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    elif architecture_name == 'resnet152':
        model = torchvision.models.resnet152(pretrained=True)
    elif architecture_name == 'densenet121':
        return torchvision.models.densenet121(pretrained=True)
    elif architecture_name == 'densenet169':
        return torchvision.models.densenet169(pretrained=True)
    elif architecture_name == 'densenet161':
        return torchvision.models.densenet161(pretrained=True)
    elif architecture_name == 'densenet201':
        return torchvision.models.densenet201(pretrained=True)
    elif architecture_name == 'inception_v3':
        return torchvision.models.inception_v3(pretrained=True)

    # Remove last layer (classification) to gain a feature extractor
    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)

    if freeze_layers:
        for p in model.parameters():
            p.requires_grad = False

    return model


def _load_img2seq_block(img2seq_config):
    """
    :param architecture_name:
    :param output_dim: Output dimensions for the transformer output
    :return:
    """
    architecture_name = img2seq_config['ARCHITECTURE']

    if architecture_name == 'transformer':
        model = Transformer(
            n_tgt_vocab=img2seq_config['D_OUTPUT'],
            len_max_seq=img2seq_config['MAX_SEQUENCE_LENGTH'],
            d_word_vec=img2seq_config['D_WORD_VEC'],
            d_model=img2seq_config['D_MODEL'],
            d_inner=img2seq_config['D_INNER'],
            n_layers=img2seq_config['N_LAYERS'],
            n_head=img2seq_config['N_HEAD'],
            d_k=img2seq_config['D_K'],
            d_v=img2seq_config['D_V'],
            dropout=img2seq_config['DROPOUT'],
            pos_enc_regularizer=img2seq_config['POSITION_ENCODER_REGULARIZER'],
        )

        return model


def load_model(config):
    use_cuda = config['CUDA'] and torch.cuda.is_available()
    model_config = config['MODEL']

    feature_block = _load_feature_block(architecture_name=model_config['FEATURE']['ARCHITECTURE'],
                                        freeze_layers=model_config['FEATURE']['FREEZE'])
    img2seq_block = _load_img2seq_block(img2seq_config=model_config['IMG_TO_SEQ'])

    output_dim = model_config['IMG_TO_SEQ']['D_OUTPUT']

    neural_modeler = NeuralModelNet(feature_extractor=feature_block,
                                    img_to_seq_block=img2seq_block,
                                    output_dim=output_dim)

    device = torch.device('cuda' if use_cuda else 'cpu')

    return neural_modeler.to(device)
