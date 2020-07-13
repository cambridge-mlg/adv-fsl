from mnasnet import film_mnasnet1_0, mnasnet1_0


def create_feature_extractor(feature_adaptation, pretrained_path):
    if feature_adaptation == "film":
        feature_extractor = film_mnasnet1_0(
            pretrained=True,
            progress=True,
            pretrained_model_path=pretrained_path,
            batch_normalization='eval'
        )
    else:  # no adaptation
        feature_extractor = mnasnet1_0(
            pretrained=True,
            progress=True,
            pretrained_model_path=pretrained_path,
            batch_normalization='eval'
        )

    # Freeze the parameters of the feature extractor
    for param in feature_extractor.parameters():
        param.requires_grad = False

    feature_extractor.eval()

    return feature_extractor
