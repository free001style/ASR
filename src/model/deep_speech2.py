import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, module):
        """
        Args:
            module (nn.Sequential): sequential of conv layers, batch norms and activations.
        """
        super(ConvBlock, self).__init__()
        self.module = module

    def forward(self, x, length):
        """
        ConvBlock forward method.

        Args:
            x (Tensor): (B , 1 , F , T) tensor of spectrogram.
            length (Tensor): (B, ) tensor of spectrogram original lengths.
        Returns:
            output (Tensor): (B , C' , F' , T') tensor -- output of ConvBlock.
            length (Tensor): (B, ) tensor -- new temporal lengths of output.
        """
        for module in self.module:
            x = module(x)
            length = self.transform_input_lengths(length, module)
            b, c, f, t = x.shape
            mask = (
                torch.arange(t).tile((b, c, f, 1)) >= length[:, None, None, None]
            ).to(x.device)
            x = x.masked_fill(mask, 0)
        return x, length

    @staticmethod
    def transform_input_lengths(length, module):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            length (Tensor): old input lengths.
            module (nn.Module): current layer.
        Returns:
            output_lengths (Tensor): new temporal lengths.
        """
        if type(module) is not nn.Conv2d:
            return length
        else:
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            return (
                length
                + 2 * module.padding[1]
                - module.dilation[1] * (module.kernel_size[1] - 1)
                - 1
            ) // module.stride[1] + 1


class RNNBlock(nn.Module):
    """
    Class for RNN layer.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        rnn_type,
        batch_first=True,
        dropout=0.0,
        bidirectional=True,
        batch_norm=True,
    ):
        """
        Args:
            input_size (int): number of input features.
            hidden_size (int): number of hidden features.
            rnn_type (str): type of rnn layer i.e. GRU, LSTM or RNN.
            batch_first (bool): whether to use batch dim first im rnn.
            dropout (float): dropout probability.
            bidirectional (bool): whether to use bidirectional or unidirectional rnn.
            batch_norm (bool): whether to use batch norm.
        """
        super(RNNBlock, self).__init__()
        self.rnn = getattr(torch.nn, rnn_type)(
            input_size,
            hidden_size,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.bn = nn.BatchNorm1d(input_size) if batch_norm else None

    def forward(self, x, length):
        """
        Args:
            x (Tensor): (B, T, F) input after convolution layers.
            length (Tensor): (B, ) transformed lengths after convolution layers.
        Returns:
            output (Tensor): (B, T, F) rnn output.
        """
        if self.bn is not None:
            x = self.bn(x.transpose(1, 2)).transpose(1, 2).contiguous()  # B x T x F
        x = nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True)
        x, _ = self.rnn(x, None)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.rnn.bidirectional:
            x = x[..., : self.rnn.hidden_size] + x[..., self.rnn.hidden_size :]
        return x


class DeepSpeech2(nn.Module):
    """
    DeepSpeech2 implementation based on https://arxiv.org/abs/1512.02595.
    """

    def __init__(
        self,
        n_feats,
        n_tokens,
        fc_hidden,
        num_rnn,
        dropout,
        rnn_type: str,
        bidirectional=True,
    ):
        """
        Args:
            n_feats (int): number of spectrogram input features (F in code notation).
            n_tokens (int): number of tokens in vocab.
            fc_hidden (int): number of hidden features (for rnn and fc).
            num_rnn (int): number of rnn layers.
            dropout (float): dropout probability.
            rnn_type (str): type of rnn layer i.e. GRU, LSTM or RNN.
            bidirectional (bool): whether to use bidirectional or unidirectional rnn
        """
        super(DeepSpeech2, self).__init__()
        if rnn_type.upper() not in ["RNN", "GRU", "LSTM"]:
            raise ValueError(f"rnn_type should be RNN, GRU or LSTM, not {rnn_type}")
        else:
            rnn_type = rnn_type.upper()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, (21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 96, (21, 11), stride=(2, 1), padding=(10, 5), bias=False),
        )
        self.conv = ConvBlock(self.conv)

        input_size = n_feats
        input_size = (input_size + 20 * 2 - 41) // 2 + 1
        input_size = (input_size + 10 * 2 - 21) // 2 + 1
        input_size = (input_size + 10 * 2 - 21) // 2 + 1

        self.gru = nn.Sequential(
            RNNBlock(
                input_size * 96,
                fc_hidden,
                rnn_type,
                True,
                dropout,
                bidirectional,
                False,
            ),
            *[
                RNNBlock(
                    fc_hidden, fc_hidden, rnn_type, True, dropout, bidirectional, True
                )
                for _ in range(num_rnn - 1)
            ],
        )
        self.bn = nn.BatchNorm1d(fc_hidden)
        self.fc = nn.Linear(fc_hidden, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Args:
            spectrogram (Tensor): (B, F, T) tensor of spectrogram.
            spectrogram_length (Tensor): (B, ) tensor of spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        spectrogram = spectrogram.unsqueeze(1)  # add channel dim
        x, length = self.conv(spectrogram, spectrogram_length)  # B x C x F x T
        b, c, f, t = x.shape
        x = x.view(b, t, c * f)
        for gru in self.gru:
            x = gru(x, length)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2).contiguous()
        logits = self.fc(x)  # B x T x n_tokens
        log_probs = F.log_softmax(logits, dim=-1)
        return {"log_probs": log_probs, "log_probs_length": length}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


# lst = [100, 99, 98]
#
# t = torch.zeros(3, 128, 110)
# for i in range(3):
#     t[i, :, :lst[i]] = torch.arange(lst[i]).expand(128, lst[i])
# print(t.shape)
#
# model = DeepSpeech2(128, 28, 768, 7, 0, nn.GRU)
# print(model)
