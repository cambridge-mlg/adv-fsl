import torch.nn as nn
from film_params import FilmLayer, FilmAdapter, NullFeatureAdaptationNetwork


class FilmClassifier(nn.Module):
    def __init__(self, num_classes, feature_extractor, feature_adaptation):
        super(FilmClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.film_adapter = self.create_film_adapter(feature_adaptation)
        self.fc = nn.Linear(in_features=self.feature_extractor.output_size, out_features=num_classes, bias=True)

    def forward(self, x):
        film_params = self.film_adapter(None)
        x = self.feature_extractor(x, film_params)
        return self.fc(x)

    def create_film_adapter(self, feature_adaptation):
        if feature_adaptation == "film":
            film_adapter = FilmAdapter(
                layer=FilmLayer,
                num_maps=[[32, 32, 16], [48, 72, 72], [72, 120, 120], [240, 480, 480], [480, 576],
                          [576, 1152, 1152, 1152], [1152], [1280]],
                num_blocks=[3, 3, 3, 3, 2, 4, 1, 1],
            )
        else:
            film_adapter = NullFeatureAdaptationNetwork()

        return film_adapter
