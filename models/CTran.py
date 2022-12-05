import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stop
from .transformer_layers import SelfAttnLayer
from .backbone import Backbone
from .utils import custom_replace, weights_init
from .position_enc import PositionEmbeddingSine, positionalencoding2d

 
class CTranModel(nn.Module):
    def __init__(self, num_labels, use_lmt, pos_emb=False, layers=3, heads=4, dropout=0.1, int_loss=0, no_x_features=False):
        super(CTranModel, self).__init__()
        self.use_lmt = use_lmt
        # for no image features)
        self.no_x_features = no_x_features

        # ResNet backbone: ResNet 101
        self.backbone = Backbone()
        hidden = 2048 # this should match the backbone output feature size

        # 下采样
        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden,hidden,(1,1))
        
        # Label Embeddings
        # 输入的标签
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1,-1).long()
        # 标签嵌入
        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None)

        # State Embeddings
        # padding_idx=0, 长度不一致时补 0 再进 EmbeddingLayer
        self.known_label_lt = torch.nn.Embedding(3, hidden, padding_idx=0)

        # 图像特征的位置编码
        # Position Embeddings (for image features)
        self.use_pos_enc = pos_emb
        if self.use_pos_enc:
            # self.position_encoding = PositionEmbeddingSine(int(hidden/2), normalize=True)
            self.position_encoding = positionalencoding2d(hidden, 18, 18).unsqueeze(0)

        # Transformer
        # 本文使用的 3 层、4 头 Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden,heads,dropout) for _ in range(layers)])

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.output_linear = torch.nn.Linear(hidden, num_labels)

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)


    def forward(self, images, mask):
        const_label_input = self.label_input.repeat(images.size(0), 1).cuda()
        init_label_embeddings = self.label_lt(const_label_input)

        # print(f"label_input:{self.label_input}")
        # # tensor([[0, 1, ..., 78, 79]])
        # print(f"const_label_input:{const_label_input}")
        # # tensor([[0, 1, ..., 78, 79]], device='cuda:0')
        # print(f"init_label_embeddings:{init_label_embeddings}")
        # # tensor([[[l1_emb1, l1_emb2, ...], [l2_emb1, ...], ..., [l79_emb1, ...]]])

        # 提取图像特征
        features = self.backbone(images)
        
        if self.downsample:
            features = self.conv_downsample(features)
        
        # 位置编码
        if self.use_pos_enc:
            pos_encoding = self.position_encoding(features, torch.zeros(features.size(0), 18, 18, dtype=torch.bool).cuda())
            features = features + pos_encoding

        # 特征向量处理
        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1) 

        if self.use_lmt:
            # Convert mask values to positive integers for nn.Embedding
            # negative -1 to 0, mask 0 to 1, positive 1 to 2
            label_feat_vec = custom_replace(mask, 0, 1, 2).long()

            # Get state embeddings
            # 三种状态，0、1、2
            state_embeddings = self.known_label_lt(label_feat_vec)

            # Add state embeddings to label embeddings
            init_label_embeddings += state_embeddings

            # print(f"mask:{mask}")
            # # tensor[[0, 0, 1, 0, -1, ...]]
            # print(f"label_feat_vec:{label_feat_vec}")
            # # tensor[[1, 1, 2, 1, 0, ...]]
            # print(f"state_embeddings:{state_embeddings}")
            # # tensor[[[emb1_1, emb1_2, ...], [emb1_1, emb1_2, ...], [emb2_1, emb2_2, ...], [emb1_1, emb1_2, ...], [emb0_1, emb0_2, ...]]]
            # print(f"init_label_embeddings:{init_label_embeddings}")
            # # 相加
        
        if self.no_x_features:
            embeddings = init_label_embeddings 
        else:
            # Concat image and label embeddings
            embeddings = torch.cat((features, init_label_embeddings), 1)

        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)        
        # why? need attns?
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings,mask=None)
            attns += attn.detach().unsqueeze(0).data

        print(f"init_label_embeddings.size():{init_label_embeddings.size()}")
        print(f"-init_label_embeddings.size(1):{-init_label_embeddings.size(1)}")

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:,-init_label_embeddings.size(1):,:]
        print(f"label_embeddings.size():{label_embeddings.size()}")
        output = self.output_linear(label_embeddings) 
        print(f"output.size():{output.size()}")
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0), 1, 1).cuda()
        output = (output * diag_mask).sum(-1)
        print(f"output.size():{output.size()}")

        return output, None, attns

