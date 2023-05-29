"""
This is the description of the deep NN currently being used.
It is a small CNN for the features with an GRU encoding of the LTL task.
The features and LTL are preprocessed by utils.format.get_obss_preprocessor(...) function:
    - In that function, I transformed the LTL tuple representation into a text representation:
    - Input:  ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
    - output: ['until', 'not', 'a', 'and', 'b', 'until', 'not', 'c', 'd']
Each of those tokens get a one-hot embedding representation by the utils.format.Vocabulary class.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import torch_ac
import copy

from gym.spaces import Box, Discrete

from gnns.graphs.GCN import *
from gnns.graphs.GNN import GNNMaker

from env_model import getEnvModel
from policy_network import PolicyNetwork

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class BasicACModel(nn.Module, torch_ac.ACModel):
    def __init__(self, env, obs_space, action_space, ignoreLTL, gnn_type, dumb_ac, freeze_ltl, args):
        super().__init__()

        # Decide which components are enabled
        self.use_progression_info = "progress_info" in obs_space
        self.use_text = not ignoreLTL and (gnn_type == "GRU" or gnn_type == "LSTM") and "text" in obs_space
        self.use_ast = not ignoreLTL and ("GCN" in gnn_type) and "text" in obs_space  # True
        self.use_trans = not ignoreLTL and ("Transformer" in gnn_type) and "text" in obs_space  # True
        self.use_dfa = not ignoreLTL and ("DFA" in gnn_type) and "text" in obs_space  # True
        self.gnn_type = gnn_type
        self.device = torch.device(args.cuda)
        self.action_space = action_space
        self.dumb_ac = dumb_ac
        self.context = False

        self.freeze_pretrained_params = freeze_ltl
        if self.freeze_pretrained_params:
            print("Freezing the LTL module.")

        self.env_model = getEnvModel(env, obs_space)


        # Resize image embedding
        self.embedding_size = self.env_model.size()  # 64
        print("embedding size:", self.embedding_size)
        if self.use_text or self.use_ast or self.use_progression_info or self.use_trans or self.use_dfa:
            self.embedding_size += args.d_out  # 96

        if self.dumb_ac:
            # Define actor's model
            self.actor = PolicyNetwork(self.embedding_size, self.action_space)

            # Define critic's model
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, 1)
            )
        else:
            # Define actor's model
            self.actor = PolicyNetwork(self.embedding_size, self.action_space, hiddens=[64, 64, 64],
                                       activation=nn.ReLU())

            # Define critic's model
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs):
        embedding = self.env_model(obs)  # shape = torch.Size([16, 64])

        embedding = torch.cat((embedding, obs.text), dim=-1)
        # print(embedding[:, -4:])

        # Actor
        dist = self.actor(embedding)

        # Critic
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value

    def load_pretrained_gnn(self, model_state):
        # We delete all keys relating to the actor/critic.
        new_model_state = model_state.copy()

        for key in model_state.keys():
            if key.find("actor") != -1 or key.find("critic") != -1:
                del new_model_state[key]

        self.load_state_dict(new_model_state, strict=False)

        if self.freeze_pretrained_params:
            target = self.text_rnn if self.gnn_type == "GRU" or self.gnn_type == "LSTM" else self.gnn

            for param in target.parameters():
                param.requires_grad = False


class ACModel(nn.Module, torch_ac.ACModel):
    def __init__(self, env, obs_space, action_space, ignoreLTL, gnn_type, dumb_ac, freeze_ltl, args):
        super().__init__()

        # Decide which components are enabled
        self.use_progression_info = "progress_info" in obs_space
        self.use_text = not ignoreLTL and (gnn_type == "GRU" or gnn_type == "LSTM") and "text" in obs_space
        self.use_ast = not ignoreLTL and ("GCN" in gnn_type) and "text" in obs_space  # True
        self.use_trans = not ignoreLTL and ("Transformer" in gnn_type) and "text" in obs_space  # True
        self.gnn_type = gnn_type
        self.device = torch.device(args.cuda)
        self.action_space = action_space
        self.dumb_ac = dumb_ac
        self.context = False

        self.freeze_pretrained_params = freeze_ltl
        if self.freeze_pretrained_params:
            print("Freezing the LTL module.")

        self.env_model = getEnvModel(env, obs_space)

        # Define text embedding
        if self.use_progression_info:
            self.text_embedding_size = 32
            self.simple_encoder = nn.Sequential(
                nn.Linear(obs_space["progress_info"], 64),
                nn.Tanh(),
                nn.Linear(64, self.text_embedding_size),
                nn.Tanh()
            ).to(self.device)
            print("Linear encoder Number of parameters:", sum(p.numel() for p in self.simple_encoder.parameters() if p.requires_grad))

        elif self.use_text:
            self.word_embedding_size = 32
            self.text_embedding_size = args.gnn_out
            if self.gnn_type == "GRU":
                self.text_rnn = GRUModel(obs_space["text"], self.word_embedding_size, 16, self.text_embedding_size).to(self.device)
            else:
                assert(self.gnn_type == "LSTM")
                self.text_rnn = LSTMModel(obs_space["text"], self.word_embedding_size, 16, self.text_embedding_size).to(self.device)
            print("RNN Number of parameters:", sum(p.numel() for p in self.text_rnn.parameters() if p.requires_grad))
        
        elif self.use_ast:
            hidden_dim = 32
            self.text_embedding_size = 32
            self.gnn = GNNMaker(self.gnn_type, obs_space["text"], self.text_embedding_size).to(self.device)
            print("GNN Number of parameters:", sum(p.numel() for p in self.gnn.parameters() if p.requires_grad))

        elif self.use_trans:
            self.word_embedding_size = 512
            self.text_embedding_size = args.d_out
            self.ltl2transformer = TransfomerSyn(obs_space["text"], self.word_embedding_size, self.text_embedding_size, 'mean' , args)
            print("Transformer Number of parameters:", sum(p.numel() for p in self.ltl2transformer.parameters() if p.requires_grad))

       # Resize image embedding
        self.embedding_size = self.env_model.size()  # 64
        print("embedding size:", self.embedding_size)
        if self.use_text or self.use_ast or self.use_progression_info or self.use_trans:
            self.embedding_size += self.text_embedding_size  # 96

        if self.dumb_ac:
            # Define actor's model
            self.actor = PolicyNetwork(self.embedding_size, self.action_space)

            # Define critic's model
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, 1)
            )
        else:
            # Define actor's model
            self.actor = PolicyNetwork(self.embedding_size, self.action_space, hiddens=[64, 64, 64], activation=nn.ReLU())

            # Define critic's model
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        # Initialize parameters correctly
        self.apply(init_params)
        if self.use_trans and args.TFixup:
            self.ltl2transformer.init_by_TFixup(args)

    def forward(self, obs):
        embedding = self.env_model(obs)  # shape = torch.Size([16, 64])

        if self.use_progression_info:
            embed_ltl = self.simple_encoder(obs.progress_info)
            embedding = torch.cat((embedding, embed_ltl), dim=1) if embedding is not None else embed_ltl

        # Adding Text
        elif self.use_text:
            embed_text = self.text_rnn(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1) if embedding is not None else embed_text

        # Adding GNN
        elif self.use_ast:
            embed_gnn = self.gnn(obs.text)  # shape = torch.Size([16, 32])
            embedding = torch.cat((embedding, embed_gnn), dim=1) if embedding is not None else embed_gnn  # shape = torch.Size([16, 96])

        elif self.use_trans:
            embed_transformer = self.ltl2transformer(obs.text)
            embedding = torch.cat((embedding, embed_transformer), dim=1) if embedding is not None else embed_transformer

        # Actor
        dist = self.actor(embedding)

        # Critic
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value

    def load_pretrained_gnn(self, model_state):
        # We delete all keys relating to the actor/critic.
        new_model_state = model_state.copy()

        for key in model_state.keys():
            if key.find("actor") != -1 or key.find("critic") != -1:  # ??? key.find()?
                del new_model_state[key]

        self.load_state_dict(new_model_state, strict=False)

        if self.freeze_pretrained_params:
            target = self.text_rnn if self.gnn_type == "GRU" or self.gnn_type == "LSTM" else self.gnn

            for param in target.parameters():
                param.requires_grad = False


class TransfomerSyn(nn.Module):
    def __init__(self, obs_size, d_model, d_out, pool, args):
        super(TransfomerSyn, self).__init__()
        self.embedded = nn.Embedding(obs_size, args.d_model)
        self.transformer = TransformerEncoderModel(d_model=args.d_model, nhead=args.nhead,
                                                   num_encoder_layers=args.num_encoder_layers,
                                                   pool=args.pool, dim_feedforward=args.dim_feedforward,
                                                   dropout=args.dropout, d_out=args.d_out,
                                                   layer_norm_eps=args.layer_norm_eps)

    def forward(self, text):
        embed_text = self.embedded(text)
        feature = self.transformer(embed_text)
        return feature

    def init_by_TFixup(self, args):  # todo:debug
        # for k, v in self.transformer.named_parameters():
        #     print(k, v, v.shape)

        for p in self.embedded.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, 0, args.d_model ** (- 1. / 2.))

        temp_state_dic = {}
        for name, param in self.embedded.named_parameters():
            if 'weight' in name:
                temp_state_dic[name] = ((9 * args.num_encoder_layers) ** (- 1. / 4.)) * param

        for name in self.embedded.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.embedded.state_dict()[name]
        self.embedded.load_state_dict(temp_state_dic)

        temp_state_dic = {}
        for name, param in self.transformer.named_parameters():
            if any(s in name for s in ["linear1.weight", "linear2.weight", "self_attn.out_proj.weight"]):
                temp_state_dic[name] = (0.67 * (args.num_encoder_layers) ** (- 1. / 4.)) * param
            elif "self_attn.in_proj_weight" in name:
                temp_state_dic[name] = (0.67 * (args.num_encoder_layers) ** (- 1. / 4.)) * (param * (2 ** 0.5))

        for name in self.transformer.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.transformer.state_dict()[name]
        self.transformer.load_state_dict(temp_state_dic)

class LSTMModel(nn.Module):
    def __init__(self, obs_size, word_embedding_size=32, hidden_dim=32, text_embedding_size=32):
        super().__init__()
        # For all our experiments we want the embedding to be a fixed size so we can "transfer". 
        self.word_embedding = nn.Embedding(obs_size, word_embedding_size)
        self.lstm = nn.LSTM(word_embedding_size, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(2*hidden_dim, text_embedding_size)

    def forward(self, text):
        hidden, _ = self.lstm(self.word_embedding(text))
        return self.output_layer(hidden[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, obs_size, word_embedding_size=32, hidden_dim=32, text_embedding_size=32):
        super().__init__()
        self.word_embedding = nn.Embedding(obs_size, word_embedding_size)
        # word_embedding_size = 32, hidden_dim = 16
        self.gru = nn.GRU(word_embedding_size, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(2*hidden_dim, text_embedding_size)

    def forward(self, text):
        # hidden_shape: [16, 9, 32]  _shape: [4, 16, 16]
        hidden, _ = self.gru(self.word_embedding(text))
        return self.output_layer(hidden[:, -1, :])

class TransformerEncoderModel(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 1, pool: str = 'mean',
                 dim_feedforward: int = 2048, dropout: float = 0.1, d_out: int = 8, activation = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False):
        """

        :param d_model: the number of expected features in the encoder/decoder inputs (default=512).
        :param nhead: the number of heads in the multiheadattention models (default=8).
        :param num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        :param dim_feedforward: the dimension of the feedforward network model (default=2048).
        :param dropout: the dropout value (default=0.1).
        :param activation: the activation function of encoder/decoder intermediate layer, can be a string
                           ("relu" or "gelu") or a unary callable. Default: relu
        :param layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        :param batch_first: If ``True``, then the input and output tensors are provided
                            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        :param norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
                           other attention and feedforward operations, otherwise after. Default: ``False``

        Examples::
            >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
            >>> src = torch.rand((10, 32, 512))
            >>> out = transformer_model(src)
        """
        super(TransformerEncoderModel, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_out)
        )

        self._reset_parameters()


    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: the sequence to the encoder (required).
            src_mask: the additive mask for the src sequence (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`, `(N, S, E)` if batch_first.
            - src_mask: :math:`(S, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            where S is the source sequence length, T is the target sequence length, N is the
        batch size, E is the feature number
        """

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        memory = memory.mean(dim=1) if self.pool == 'mean' else memory[:, -1, :]
        memory = self.to_latent(memory)
        memory = torch.tanh(self.mlp_head(memory))
        return memory

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        r"""TransformerEncoder is a stack of N encoder layers

        Args:
            encoder_layer: an instance of the TransformerEncoderLayer() class (required).
            num_layers: the number of sub-encoder-layers in the encoder (required).
            norm: the layer normalization component (optional).

        Examples::
            >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
            >>> src = torch.rand(10, 32, 512)
            >>> out = transformer_encoder(src)
        """
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self,  d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False):
        """
        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            activation: the activation function of the intermediate layer, can be a string
                        ("relu" or "gelu") or a unary callable. Default: relu
            layer_norm_eps: the eps value in layer normalization components (default=1e-5).
            batch_first: If ``True``, then the input and output tensors are provided
                         as (batch, seq, feature). Default: ``False``.
            norm_first: if ``True``, layer norm is done prior to attention and feedforward
                        operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
        Examples::
            >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
            >>> src = torch.rand(32, 10, 512)
            >>> out = encoder_layer(src)
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

