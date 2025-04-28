import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        # здесь device берём из входного тензора
        device = input_char.device
        batch_size = input_char.size(0)
        # создаём сразу на нужном девайсе
        one_hot = torch.zeros(batch_size, onehot_dim, device=device, dtype=torch.float)
        # проводим scatter
        return one_hot.scatter_(1, input_char.unsqueeze(1), 1.0)

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1

        device = batch_H.device
        # сразу создаём тензоры на том же device
        output_hiddens = torch.zeros(
            batch_size, num_steps, self.hidden_size, device=device
        )
        hidden = (
            torch.zeros(batch_size, self.hidden_size, device=device),
            torch.zeros(batch_size, self.hidden_size, device=device),
        )

        if is_train:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    text[:, i], onehot_dim=self.num_classes
                )
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]
            probs = self.generator(output_hiddens)
        else:
            targets = torch.zeros(batch_size, dtype=torch.long, device=device)
            probs = torch.zeros(batch_size, num_steps, self.num_classes, device=device)
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes
                )
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs


class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # batch_H: [B x T x C]
        device = batch_H.device
        batch_H_proj = self.i2h(batch_H)  # [B x T x H]
        prev_h_proj = self.h2h(prev_hidden[0]).unsqueeze(1)  # [B x 1 x H]
        e = self.score(torch.tanh(batch_H_proj + prev_h_proj))  # [B x T x 1]

        alpha = F.softmax(e, dim=1)  # [B x T x 1]
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # [B x C]
        concat = torch.cat([context, char_onehots], dim=1)  # [B x (C+E)]
        cur_hidden = self.rnn(concat, prev_hidden)
        return cur_hidden, alpha
