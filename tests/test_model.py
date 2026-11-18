"""Check NACFN training and inference with synthetic inputs."""

import torch
from torch import nn

from models.model import NACFN, IntraDayNutritionEncoder, StaticEncoder


def test_static_encoder_pools_with_sparsemax():
    encoder = StaticEncoder(num_features=3, dim=4, heads=2, layers=1, dropout=0)
    encoder.tokenizer = nn.Identity()
    encoder.encoder = nn.Identity()
    with torch.no_grad():
        encoder.query.copy_(torch.tensor([2.0, 0.0, 0.0, 0.0]))

    tokens = torch.tensor([[[0.75, 1.0, 0.0, 0.0], [0.25, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]]])
    pooled, weights = encoder(tokens)
    torch.testing.assert_close(weights, torch.tensor([[0.75, 0.25, 0.0]]))
    torch.testing.assert_close(pooled, 0.75 * tokens[:, 0] + 0.25 * tokens[:, 1])


def test_nutrient_attention_encodes_days_independently():
    encoder = IntraDayNutritionEncoder(input_dim=4, dim=8, heads=2, layers=1, max_length=3, dropout=0)
    encoder.eval()
    values = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    changed = values.clone()
    changed[0, 1, 0] += 10
    changed[0, 2] = float("nan")
    with torch.no_grad():
        before = encoder(values, mask)
        after = encoder(changed, mask)

    assert not torch.allclose(after[0, 1], before[0, 1])
    torch.testing.assert_close(after[0, 0], before[0, 0])
    torch.testing.assert_close(after[0, 2], torch.zeros(8))
    torch.testing.assert_close(after[1], before[1])


def test_forward_backward_and_explanations():
    model = NACFN(static_dim=5, nutrition_dim=6, hidden_dim=16, num_heads=4, num_layers=1, sequence_length=4)
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
    model = NACFN(static_dim=4, nutrition_dim=3, hidden_dim=8, num_heads=2, num_layers=1, sequence_length=2).eval()
    result = model(
        torch.arange(2),
        torch.randn(2, 4),
        torch.randn(2, 2, 4),
        torch.ones(2, 2, dtype=torch.bool),
        return_details=True,
    )
    assert result.loss is None


def test_empty_nutrition_history_is_ignored():
    model = NACFN(static_dim=4, nutrition_dim=3, hidden_dim=8, num_heads=2, num_layers=1, sequence_length=2).eval()
    sample_id = torch.arange(2)
    static = torch.randn(2, 4)
    mask = torch.tensor([[False, False], [True, False]])
    nutrition = torch.randn(2, 2, 4)
    with torch.no_grad():
        expected = model(sample_id, static, nutrition, mask, return_details=True)
        output = model(
            sample_id, static, nutrition.masked_fill(~mask.unsqueeze(-1), float("nan")), mask, return_details=True
        )

    assert torch.isfinite(output.prediction).all()
    torch.testing.assert_close(output.prediction, expected.prediction)
    torch.testing.assert_close(output.temporal_attention, torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    assert output.history_gate[0] == 0
