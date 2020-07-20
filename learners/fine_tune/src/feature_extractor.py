from mnasnet import film_mnasnet1_0, mnasnet1_0
from resnet import film_resnet18, resnet18


def create_feature_extractor(feature_extractor, feature_adaptation, pretrained_path):
    if feature_adaptation == "film":
        if feature_extractor == "mnasnet":
            feature_extractor = film_mnasnet1_0(
                pretrained=True,
                progress=True,
                pretrained_model_path=pretrained_path,
                batch_normalization='eval'
            )
        else:
            feature_extractor = film_resnet18(
                pretrained=True,
                pretrained_model_path=pretrained_path,
                batch_normalization='eval'
            )
    else:  # no adaptation
        if feature_extractor == "mnasnet":
            feature_extractor = mnasnet1_0(
                pretrained=True,
                progress=True,
                pretrained_model_path=pretrained_path,
                batch_normalization='eval'
            )
        else:
            feature_extractor = resnet18(
                pretrained=True,
                pretrained_model_path=pretrained_path,
                batch_normalization='eval'
            )

    # Freeze the parameters of the feature extractor
    for param in feature_extractor.parameters():
        param.requires_grad = False

    feature_extractor.eval()

    return feature_extractor
