# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import numpy as np


class MLPLayer(nn.Layer):
    def __init__(self, input_shape, units_list=None, l2=0.01, last_action=None, **kwargs):
        super(MLPLayer, self).__init__(**kwargs)

        if units_list is None:
            units_list = [128, 128, 64]
        units_list = [input_shape] + units_list

        self.units_list = units_list
        self.l2 = l2
        self.mlp = []
        self.last_action = last_action

        for i, unit in enumerate(units_list[:-1]):
            if i != len(units_list) - 1:
                dense = paddle.nn.Linear(in_features=unit,
                                         out_features=units_list[i + 1],
                                         weight_attr=paddle.ParamAttr(
                                             initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(unit))))
                self.mlp.append(dense)
                self.add_sublayer('dense_%d' % i, dense)

                relu = paddle.nn.ReLU()
                self.mlp.append(relu)
                self.add_sublayer('relu_%d' % i, relu)

                norm = paddle.nn.BatchNorm1D(units_list[i + 1])
                self.mlp.append(norm)
                self.add_sublayer('norm_%d' % i, norm)
            else:
                dense = paddle.nn.Linear(in_features=unit,
                                         out_features=units_list[i + 1],
                                         weight_attr=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(unit)))
                self.mlp.append(dense)
                self.add_sublayer('dense_%d' % i, dense)

                if last_action is not None:
                    relu = paddle.nn.ReLU()
                    self.mlp.append(relu)
                    self.add_sublayer('relu_%d' % i, relu)

    def forward(self, inputs):
        outputs = inputs
        for n_layer in self.mlp:
            outputs = n_layer(outputs)
        return outputs


class FFMLayer(nn.Layer):
    def __init__(self,
                 sparse_feature_number,
                 sparse_feature_dim,
                 dense_feature_dim,
                 sparse_num_field,
                 reduction,
                 dnn_layers_size):
        super(FFMLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.reduction = reduction
        self.dnn_layers_size = dnn_layers_size

        # self.ffm = FFM(sparse_feature_number, sparse_feature_dim,
        #                dense_feature_dim, sparse_num_field)

        self.ffm = DeepFFM(
            sparse_feature_number=sparse_feature_number,
            sparse_feature_dim=self.sparse_feature_dim,
            dense_feature_dim=self.dense_feature_dim,
            sparse_num_field=self.sparse_num_field,
            reduction=self.reduction,
            dnn_layers_size=self.dnn_layers_size
        )

        self.bias = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, sparse_inputs, dense_inputs):

        y_first_order, y_second_order = self.ffm(sparse_inputs, dense_inputs)

        predict = F.sigmoid(y_first_order + y_second_order + self.bias)

        return predict


class CENet(nn.Layer):
    def __init__(self,
                 sparse_field_num,
                 reduction):
        super(CENet, self).__init__()
        self.sparse_field_num = sparse_field_num ** 2
        self.reduced_num_fields = self.sparse_field_num // reduction

        self.pooling = paddle.nn.AdaptiveAvgPool1D(output_size=1)
        self.mlp = MLPLayer(input_shape=self.sparse_field_num,
                            units_list=[self.reduced_num_fields, self.sparse_field_num],
                            last_action="relu")

    def forward(self, inputs):
        """
        Forward calculation of ComposeExcitationNetworkLayer

        Args:
            inputs, shape = (batch_size, sparse_num_field, sparse_num_field * embedding_size)
        """
        B, N, N_E = paddle.shape(inputs)
        inputs = paddle.reshape(inputs, shape=[B, N**2, -1])
        # (B, N^2, 1)
        pooled_inputs = self.pooling(inputs)

        # (B, N^2)
        pooled_inputs = paddle.squeeze(pooled_inputs, axis=-1)

        # (B, N^2)
        attn_w = self.mlp(pooled_inputs)

        # (B, N^2, E)
        outputs = paddle.multiply(inputs, paddle.unsqueeze(attn_w, axis=-1))
        return paddle.reshape(outputs, shape=[B, N, -1])


class FFM(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field):
        super(FFM, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.dense_emb_dim = self.sparse_feature_dim
        self.sparse_num_field = sparse_num_field
        self.init_value_ = 0.1

        # sparse part coding
        self.embedding_one = paddle.nn.Embedding(
            sparse_feature_number,
            1,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim * self.sparse_num_field,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        # dense part coding
        self.dense_w_one = paddle.create_parameter(
            shape=[self.dense_feature_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0))

        self.dense_w = paddle.create_parameter(
            shape=[
                1, self.dense_feature_dim,
                self.dense_emb_dim * self.sparse_num_field
            ],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0))

    def forward(self, sparse_inputs, dense_inputs):
        # -------------------- first order term  --------------------
        sparse_inputs_concat = paddle.concat(sparse_inputs, axis=1)
        sparse_emb_one = self.embedding_one(sparse_inputs_concat)

        dense_emb_one = paddle.multiply(dense_inputs, self.dense_w_one)
        dense_emb_one = paddle.unsqueeze(dense_emb_one, axis=2)

        y_first_order = paddle.sum(sparse_emb_one, 1) + paddle.sum(
            dense_emb_one, 1)

        # -------------------Field-aware second order term  --------------------
        sparse_embeddings = self.embedding(sparse_inputs_concat)
        dense_inputs_re = paddle.unsqueeze(dense_inputs, axis=2)
        dense_embeddings = paddle.multiply(dense_inputs_re, self.dense_w)
        feat_embeddings = paddle.concat([sparse_embeddings, dense_embeddings],
                                        1)

        field_aware_feat_embedding = paddle.reshape(
            feat_embeddings,
            shape=[
                -1, self.sparse_num_field, self.sparse_num_field,
                self.sparse_feature_dim
            ])
        field_aware_interaction_list = []
        for i in range(self.sparse_num_field):
            for j in range(i + 1, self.sparse_num_field):
                field_aware_interaction_list.append(
                    paddle.sum(field_aware_feat_embedding[:, i, j, :] *
                               field_aware_feat_embedding[:, j, i, :],
                               1,
                               keepdim=True))

        y_field_aware_second_order = paddle.add_n(field_aware_interaction_list)
        return y_first_order, y_field_aware_second_order


class DeepFFM(nn.Layer):
    def __init__(self,
                 sparse_feature_number,
                 sparse_feature_dim,
                 dense_feature_dim,
                 sparse_num_field,
                 reduction,
                 dnn_layers_size):
        super(DeepFFM, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.dense_emb_dim = self.sparse_feature_dim
        self.sparse_num_field = sparse_num_field
        self.init_value_ = 0.1
        self.dnn_layers_size = dnn_layers_size
        self.reduction = reduction

        # sparse part coding
        self.embedding_one = paddle.nn.Embedding(
            sparse_feature_number,
            1,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim * self.sparse_num_field,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        # dense part coding
        self.dense_w_one = paddle.create_parameter(
            shape=[self.dense_feature_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0))

        self.dense_w = paddle.create_parameter(
            shape=[
                1, self.dense_feature_dim,
                self.dense_emb_dim * self.sparse_num_field
            ],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0))

        self.cen = CENet(self.sparse_num_field, self.reduction)

        self.mlp = MLPLayer(input_shape=int(self.sparse_num_field * (self.sparse_num_field - 1) / 2),
                            units_list=dnn_layers_size,
                            last_action="relu")

    def forward(self, sparse_inputs, dense_inputs):
        # -------------------- first order term  --------------------
        sparse_inputs_concat = paddle.concat(sparse_inputs, axis=1)
        sparse_emb_one = self.embedding_one(sparse_inputs_concat)

        dense_emb_one = paddle.multiply(dense_inputs, self.dense_w_one)
        dense_emb_one = paddle.unsqueeze(dense_emb_one, axis=2)

        y_first_order = paddle.sum(sparse_emb_one, 1) + paddle.sum(
            dense_emb_one, 1)

        # -------------------Field-aware second order term  --------------------
        sparse_embeddings = self.embedding(sparse_inputs_concat)
        dense_inputs_re = paddle.unsqueeze(dense_inputs, axis=2)
        dense_embeddings = paddle.multiply(dense_inputs_re, self.dense_w)
        # (batch_size, sparse_num_field, sparse_num_field * embedding_size)
        feat_embeddings = paddle.concat([sparse_embeddings, dense_embeddings],
                                        1)

        feat_embeddings = self.cen(feat_embeddings)

        field_aware_feat_embedding = paddle.reshape(
            feat_embeddings,
            shape=[
                -1, self.sparse_num_field, self.sparse_num_field,
                self.sparse_feature_dim
            ])
        field_aware_interaction_list = []
        for i in range(self.sparse_num_field):
            for j in range(i + 1, self.sparse_num_field):
                field_aware_interaction_list.append(
                    paddle.sum(field_aware_feat_embedding[:, i, j, :] *
                               field_aware_feat_embedding[:, j, i, :],
                               1,
                               keepdim=True))

        # y_field_aware_second_order = paddle.add_n(field_aware_interaction_list)
        ffm_output = paddle.concat(field_aware_interaction_list, axis=1)
        y_field_aware_second_order = self.mlp(ffm_output)

        return y_first_order, y_field_aware_second_order
