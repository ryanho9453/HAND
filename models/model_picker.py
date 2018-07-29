from models.conv_model import ConvModel


class ModelPicker:
    def __init__(self, config):
        self.config = config
        self.model_name = config['choose_model']
        self.model_param = config[self.model_name]

    def pick_model(self):
        """
        add if/ else if other model is used
        """
        return ConvModel(self.config, self.model_param)

