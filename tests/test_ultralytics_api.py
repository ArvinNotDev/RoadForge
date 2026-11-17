import pytest


@pytest.mark.integration
def test_installed_yolo26_semantic_model_builds_and_uses_four_class_head():
    ultralytics = pytest.importorskip("ultralytics")
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import SemanticSegmentationModel

    model = YOLO("yolo26n-sem.yaml", task="semantic", verbose=False)
    assert ultralytics.__version__ == "8.4.92"
    assert model.task == "semantic"
    assert isinstance(model.model, SemanticSegmentationModel)
    adapted = SemanticSegmentationModel(model.model.yaml, nc=4, ch=3, verbose=False)
    assert adapted.yaml["nc"] == 4
    image = torch.zeros(1, 3, 128, 128)
    adapted.eval()
    with torch.inference_mode():
        output = adapted(image)
    if isinstance(output, (tuple, list)):
        output = output[0]
    assert output.shape[1] == 4
