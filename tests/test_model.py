import torch
from models.model import FBGPredictor

def test_forward_backward_and_explanations():
    model = FBGPredictor(static_dim=5, nutrition_dim=6, hidden_dim=16, num_heads=4, num_layers=1, sequence_length=4)
    sample_id = torch.arange(3)
    static = torch.randn(3, 5)
    nutrition = torch.randn(3, 4, 7)
    mask = torch.tensor([[False, True, True, True], [False, False, True, True], [True, True, True, True]])
    target = torch.randn(3)
    output = model(sample_id, static, nutrition, mask, target, return_details=True)
    assert output.prediction.shape == (3,)
    assert output.static_attention.shape == (3, 5)
    assert output.temporal_attention.shape == (3, 4)
    assert torch.allclose(output.static_attention.sum(1), torch.ones(3), atol=1e-5)
    output.loss.backward()
    assert any(p.grad is not None for p in model.parameters())

def test_inference_without_target():
    model = FBGPredictor(static_dim=4, nutrition_dim=3, hidden_dim=8, num_heads=2, num_layers=1, sequence_length=2).eval()
    result = model(torch.arange(2), torch.randn(2, 4), torch.randn(2, 2, 4), torch.ones(2, 2, dtype=torch.bool), return_details=True)
    assert result.loss is None
