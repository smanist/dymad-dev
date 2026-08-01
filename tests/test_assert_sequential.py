import torch

from dymad.modules.sequential import SimpleRNN


def test_simple_rnn_hidden_state_follows_input_device() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleRNN(
        seq_len=2,
        input_dim=4,
        hidden_dim=2,
        output_dim=2,
        n_layers=1,
        dtype=torch.float64,
    )
    model.to(device)
    inputs = torch.ones((3, 4), dtype=torch.float64, device=device)

    output = model(inputs)

    assert output.device == device
