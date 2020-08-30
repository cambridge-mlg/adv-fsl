import torch
from learners.fine_tune.src.mnasnet import mnasnet1_0
from learners.fine_tune.src.resnet import resnet18
from learners.fine_tune.src.convnet import ConvnetFeatureExtractor


def create_feature_extractor(feature_extractor_family, pretrained_path):
    if feature_extractor_family == "mnasnet":
        feature_extractor = mnasnet1_0(
            pretrained=True,
            progress=True,
            pretrained_model_path=pretrained_path,
            batch_normalization='eval'
        )

    elif feature_extractor_family == "resnet":
        feature_extractor = resnet18(
            pretrained=True,
            pretrained_model_path=pretrained_path,
            batch_normalization='eval'
        )

    elif feature_extractor_family == "maml_convnet":
        feature_extractor = ConvnetFeatureExtractor(3, 32)
        saved_model_dict = torch.load(pretrained_path)
        feature_extractor.load_state_dict(saved_model_dict['state_dict'])

    elif feature_extractor_family == "protonets_convnet":
        feature_extractor = ConvnetFeatureExtractor(3, 64)
        saved_model_dict = torch.load(pretrained_path)
        feature_extractor.load_state_dict(saved_model_dict['state_dict'])

    else:
        feature_extractor = None

    return feature_extractor
