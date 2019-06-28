import torch
from src.data.data_utils import create_dataloader
from inference.neural_modeller import NeuralModeller
from inference.naive_neural_modeller import NaiveNeuralModeller


class Inference:

    def __init__(self, config, engine):
        self.config = config
        assert engine is not None, 'Rendering engine is required for inference mode'
        self.engine = engine

        model = self.load_model(config)

        postprocess_type = config['TEST']['POSTPROCESS']['TYPE']
        if postprocess_type == "beam":
            self.neural_modeller = NeuralModeller(model)
        elif postprocess_type == "naive":
            self.neural_modeller = NaiveNeuralModeller(model)

    @staticmethod
    def load_model(config):

        model_path = config['MODEL']['PATH']
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()

        return model

    def infer(self):
        test_dataloader = create_dataloader(config=self.config, data_type='test', engine=self.engine)

        for i, batch in enumerate(test_dataloader):

            rendered_triplet, modifiers = batch
            batch_hyp, batch_scores = self.neural_modeller.process(rendered_triplet, limit=modifiers.shape[1])

            print(batch_hyp, batch_scores)


