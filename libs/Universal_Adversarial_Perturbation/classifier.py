import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, num_classes, feature_extractor):
        super(Classifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(in_features=self.feature_extractor.output_size, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.fc(x)
