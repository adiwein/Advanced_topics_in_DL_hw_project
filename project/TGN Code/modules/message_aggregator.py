from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np


WIKIPEDIA_MSG_SIZE = 688
MLP_LAYER_SIZES = [6, 3, 2]


class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
    self.device = device

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids, and a list of messages of the same length, aggregate different
    messages for the same id using one of the possible strategies.
    :param node_ids: A list of node ids of length batch_size
    :param messages: A tensor of shape [batch_size, message_length]
    :param timestamps A tensor of shape [batch_size]
    :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
    """

  def group_by_id(self, node_ids, messages, timestamps):
    node_id_to_messages = defaultdict(list)

    for i, node_id in enumerate(node_ids):
      node_id_to_messages[node_id].append((messages[i], timestamps[i]))

    return node_id_to_messages


class LastMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(LastMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []

    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            unique_messages.append(messages[node_id][-1][0])
            unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(MeanMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class ExtendedMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(ExtendedMessageAggregator, self).__init__(device)

  def _node_aggregate(self, node_messages):
    raise NotImplementedError()

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        unique_messages.append(self._node_aggregate(torch.stack([m[0] for m in messages[node_id]])))
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class KivMeanMessageAggregator(ExtendedMessageAggregator):
  def __init__(self, device):
    super(KivMeanMessageAggregator, self).__init__(device)

  def _node_aggregate(self, node_messages):
    return torch.mean(node_messages, dim=0)


class MLPMessageAggregator(ExtendedMessageAggregator):
  def __init__(self, device):
    super(MLPMessageAggregator, self).__init__(device)
    self.mlp = nn.Sequential(
      nn.Linear(in_features=MLP_LAYER_SIZES[0], out_features=MLP_LAYER_SIZES[1]),
      nn.ReLU(),
      nn.Linear(in_features=MLP_LAYER_SIZES[1], out_features=MLP_LAYER_SIZES[2]),
      nn.ReLU(),
      nn.Linear(in_features=MLP_LAYER_SIZES[2], out_features=1)
    )

  def _node_aggregate(self, node_messages):
    mlp_input_size = MLP_LAYER_SIZES[0]

    # if too few messages - pad with zeros
    if node_messages.shape[0] < mlp_input_size:
        node_messages = torch.cat((
            torch.zeros((mlp_input_size - node_messages.shape[0], WIKIPEDIA_MSG_SIZE), device=self.device),
            node_messages
        ), dim=0)

    node_messages = node_messages[-mlp_input_size:,:]  # crop down to last few messages

    return self.mlp(node_messages.T).squeeze() # apply per-element instead of per-message via double transpose


class MsgAdaptiveMLPMessageAggregator(ExtendedMessageAggregator):
  def __init__(self, device):
    super(MsgAdaptiveMLPMessageAggregator, self).__init__(device)
    self.mlp = nn.Sequential(
      nn.Linear(in_features=MLP_LAYER_SIZES[0] * WIKIPEDIA_MSG_SIZE, out_features=MLP_LAYER_SIZES[1] * WIKIPEDIA_MSG_SIZE),
      nn.ReLU(),
      nn.Linear(in_features=MLP_LAYER_SIZES[1] * WIKIPEDIA_MSG_SIZE, out_features=MLP_LAYER_SIZES[2] * WIKIPEDIA_MSG_SIZE),
      nn.ReLU(),
      nn.Linear(in_features=MLP_LAYER_SIZES[2] * WIKIPEDIA_MSG_SIZE, out_features=WIKIPEDIA_MSG_SIZE)
    )

  def _node_aggregate(self, node_messages):
    mlp_input_size = MLP_LAYER_SIZES[0] * WIKIPEDIA_MSG_SIZE
    node_messages = node_messages.flatten()

    # if too few messages - pad with zeros
    if node_messages.shape[0] < mlp_input_size:
        node_messages = torch.cat((
            torch.zeros(mlp_input_size - node_messages.shape[0], device=self.device),
            node_messages
        ))

    node_messages = node_messages[-mlp_input_size:]  # crop down to last few messages

    return self.mlp(node_messages).squeeze()


class ConvMessageAggregator(ExtendedMessageAggregator):
  def __init__(self, device):
    super(ConvMessageAggregator, self).__init__(device)
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,1), dilation=(2,1)),
      nn.ReLU(),
      nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,1), dilation=(2,1)),
      nn.ReLU()
    )
    self.mlp = nn.Sequential(
      nn.Linear(in_features=MLP_LAYER_SIZES[0], out_features=1),
      nn.ReLU(),
    )


  def _node_aggregate(self, node_messages):
    # if too few messages - pad with zeros
    receptive_field = 2*2+1 # dilation * n_layers + 1
    conv_min_input_size = receptive_field + (2-1) # receptive_field + (dilation - 1)
    if node_messages.shape[0] < conv_min_input_size:
        node_messages = torch.cat((
            torch.zeros((conv_min_input_size - node_messages.shape[0], WIKIPEDIA_MSG_SIZE), device=self.device),
            node_messages
        ), dim=0)

    features = self.conv(node_messages.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    mlp_input_size = MLP_LAYER_SIZES[0]

    # if too few messages - pad with zeros
    if features.shape[0] < mlp_input_size:
        features = torch.cat((
            torch.zeros((mlp_input_size - features.shape[0], WIKIPEDIA_MSG_SIZE), device=self.device),
            features
        ), dim=0)

    features = features[-mlp_input_size:,:]  # crop down to last few messages
    return self.mlp(features.T).squeeze()  # apply per-element via double transpose


class GRUMessageAggregator(ExtendedMessageAggregator):
  def __init__(self, device):
    super(GRUMessageAggregator, self).__init__(device)
    self.gru = nn.GRUCell(input_size=WIKIPEDIA_MSG_SIZE, hidden_size=WIKIPEDIA_MSG_SIZE)

  def _node_aggregate(self, node_messages):
    aggregate_msg = node_messages[0].unsqueeze(0)
    for message in node_messages:
        aggregate_msg = self.gru(message.unsqueeze(0), aggregate_msg)
    return aggregate_msg.squeeze()


class LSTMMessageAggregator(ExtendedMessageAggregator):
  def __init__(self, device):
    super(LSTMMessageAggregator, self).__init__(device)
    self.lstm = nn.LSTMCell(input_size=WIKIPEDIA_MSG_SIZE, hidden_size=WIKIPEDIA_MSG_SIZE)

  def _node_aggregate(self, node_messages):
    aggregate_msg = node_messages[0].unsqueeze(0)
    hidden_state = node_messages[0].unsqueeze(0)
    for message in node_messages:
        hidden_state, aggregate_msg = self.lstm(message.unsqueeze(0), (hidden_state, aggregate_msg))
    return aggregate_msg.squeeze()

class AttentionMessageAggregator(ExtendedMessageAggregator):
    def __init__(self, device):
        super(AttentionMessageAggregator, self).__init__(device)
        self.attn = nn.MultiheadAttention(embed_dim=WIKIPEDIA_MSG_SIZE, num_heads=1)

    def _node_aggregate(self, node_messages):
        node_messages = node_messages.unsqueeze(1)
        weighted_messages = self.attn(query=node_messages, key=node_messages, value=node_messages)[0].squeeze(1)
        return weighted_messages.sum(0)


class GRUAttentionMessageAggregator(ExtendedMessageAggregator):
    def __init__(self, device):
        super(GRUAttentionMessageAggregator, self).__init__(device)
        self.attn = nn.MultiheadAttention(embed_dim=WIKIPEDIA_MSG_SIZE, num_heads=1)
        self.gru = nn.GRUCell(input_size=WIKIPEDIA_MSG_SIZE, hidden_size=WIKIPEDIA_MSG_SIZE)

    def _node_aggregate(self, node_messages):
        aggregate_msg = node_messages[0].unsqueeze(0)
        for massage in node_messages:
            gru_weights = self.gru(massage.unsqueeze(0), aggregate_msg)
        gru_weights = gru_weights.unsqueeze(1)
        weighted_messages = self.attn(query=gru_weights, key=gru_weights, value=gru_weights)[0].squeeze(1)
        return weighted_messages.sum(0)


class LSTMAttentionMessageAggregator(ExtendedMessageAggregator):
    def __init__(self, device):
        super(LSTMAttentionMessageAggregator, self).__init__(device)
        self.attn = nn.MultiheadAttention(embed_dim=WIKIPEDIA_MSG_SIZE, num_heads=1)
        self.lstm = nn.LSTMCell(input_size=WIKIPEDIA_MSG_SIZE, hidden_size=WIKIPEDIA_MSG_SIZE)

    def _node_aggregate(self, node_messages):
        aggregate_msg = node_messages[0].unsqueeze(0)
        hidden_state = node_messages[0].unsqueeze(0)
        for massage in node_messages:
            hidden_state, lstm_weights = self.lstm(massage.unsqueeze(0), (hidden_state, aggregate_msg))
        lstm_weights = lstm_weights.unsqueeze(1)
        weighted_messages = self.attn(query=lstm_weights, key=lstm_weights, value=lstm_weights)[0].squeeze(1)
        return weighted_messages.sum(0)


def get_message_aggregator(aggregator_type, device):
  if aggregator_type == "last":
    return LastMessageAggregator(device=device)
  elif aggregator_type == "mean":
    return MeanMessageAggregator(device=device)
  elif aggregator_type == "MLP":
    return MLPMessageAggregator(device=device)
  elif aggregator_type == "messageAdaptiveMLP":
    return MsgAdaptiveMLPMessageAggregator(device=device)
  elif aggregator_type == "conv":
    return ConvMessageAggregator(device=device)
  elif aggregator_type == "GRU":
    return GRUMessageAggregator(device=device)
  elif aggregator_type == "LSTM":
    return LSTMMessageAggregator(device=device)
  elif aggregator_type == "AttMsg":
    return AttentionMessageAggregator(device=device)
  elif aggregator_type == "GRUAttMsg":
    return GRUAttentionMessageAggregator(device=device)
  elif aggregator_type == "LSTMAttMsg":
    return LSTMAttentionMessageAggregator(device=device)
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
