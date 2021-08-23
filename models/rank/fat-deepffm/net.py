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
import paddle.nn.functional as Fun
import math


class MLPLayer(nn.Layer):
    def __init__(self,
                 input_shape,
                 units_list=None,
                 l2=0.01,
                 last_action=None,
                 dropout=0.5,
                 **kwargs):
        super(MLPLayer, self).__init__(**kwargs)

        if units_list is None:
            units_list = [128, 128, 64]
        units_list = [input_shape] + units_list

        self.units_list = units_list
        self.l2 = l2
        self.mlp = []
        self.last_action = last_action
        self.dropout = dropout

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


class FAT_DeepFFMLayer(nn.Layer):
    def __init__(self,
                 sparse_feature_number,
                 sparse_feature_dim,
                 dense_feature_dim,
                 sparse_num_field,
                 reduction,
                 dnn_layers_size,
                 dense_dnn_layers_size):
        super(FAT_DeepFFMLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.reduction = reduction
        self.dnn_layers_size = dnn_layers_size
        self.dense_dnn_layers_size = dense_dnn_layers_size

        self.deepFFM = DeepFFM(
            sparse_feature_number=self.sparse_feature_number,
            sparse_feature_dim=self.sparse_feature_dim,
            dense_feature_dim=self.dense_feature_dim,
            sparse_num_field=self.sparse_num_field,
            reduction=self.reduction,
            dnn_layers_size=self.dnn_layers_size,
            dense_dnn_layers_size=self.dense_dnn_layers_size,
            is_hadamard_product=False
        )

        self.bias = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, sparse_inputs, dense_inputs):

        y_first_order, y_second_order = self.deepFFM(sparse_inputs, dense_inputs)

        predict = Fun.sigmoid(y_first_order + y_second_order + self.bias)

        return predict


class CENet(nn.Layer):
    def __init__(self,
                 sparse_field_num,
                 sparse_feature_dim,
                 reduction):
        super(CENet, self).__init__()
        self.sparse_field_num = sparse_field_num ** 2
        self.sparse_feature_dim = sparse_feature_dim
        self.reduced_num_fields = self.sparse_field_num // reduction

        self.mlp = MLPLayer(input_shape=self.sparse_field_num,
                            units_list=[self.reduced_num_fields, self.sparse_field_num],
                            last_action="relu")
        self.conv1d = paddle.create_parameter(
                            shape=[self.sparse_feature_dim * sparse_field_num, sparse_field_num],
                            dtype='float32',
                            default_initializer=paddle.nn.initializer.KaimingUniform())

    def forward(self, inputs):
        """
        Forward calculation of ComposeExcitationNetworkLayer

        Args:
            inputs, shape = (batch_size, sparse_num_field, sparse_num_field * embedding_size)
        """
        # (b, n, n*k)
        B, N, N_E = paddle.shape(inputs)

        # (b, n, n) <- (b, n, n*k) * (n*k, n)
        d_v = paddle.matmul(inputs, self.conv1d)
        d_v = Fun.relu(d_v)

        # (b, n*n)
        D = paddle.reshape(d_v, shape=[B, -1])

        # (b, n*n)
        s = self.mlp(D)

        # (b, n, n*k) <- (b, n, n, k) * (b, n, n, 1)
        aem = paddle.reshape(paddle.multiply(paddle.reshape(inputs, shape=(B, N, N, -1)),
                                             paddle.reshape(s, shape=(B, N, N, -1))),
                             shape=(B, N, -1))

        return aem


class DeepFFM(nn.Layer):
    def __init__(self,
                 sparse_feature_number,
                 sparse_feature_dim,
                 dense_feature_dim,
                 sparse_num_field,
                 reduction,
                 dnn_layers_size,
                 dense_dnn_layers_size,
                 is_hadamard_product=True):
        super(DeepFFM, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.dense_emb_dim = self.sparse_feature_dim
        self.sparse_num_field = sparse_num_field
        self.init_value_ = 0.1
        self.dnn_layers_size = dnn_layers_size
        self.dense_dnn_layers_size = dense_dnn_layers_size
        self.reduction = reduction
        self.is_hadamard_product = is_hadamard_product

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
            default_initializer=paddle.nn.initializer.KaimingUniform())

        self.cen = CENet(self.sparse_num_field, self.sparse_feature_dim, self.reduction)

        input_shape = int(self.sparse_num_field * (self.sparse_num_field - 1) / 2)
        input_shape = input_shape * self.sparse_feature_dim if self.is_hadamard_product else input_shape * 1
        self.mlp = MLPLayer(input_shape=input_shape,
                            units_list=dnn_layers_size,
                            last_action="relu")
        self.dense_dnn = MLPLayer(input_shape=self.dense_feature_dim,
                                  units_list=self.dense_dnn_layers_size,
                                  last_action="relu")

    def forward(self, sparse_inputs, dense_inputs):
        """
        sparse_inputs: (batch_size, sparse_field_num)
        dense_inputs: (batch_size, dense_feature_dim)
        """
        # -------------------- first order term  --------------------
        # (batch_size, sparse_field_num - 1, 1)
        sparse_inputs_concat = paddle.concat(sparse_inputs, axis=1)
        sparse_emb_one = self.embedding_one(sparse_inputs_concat)

        # (batch_size, dense_feature_dim)
        dense_emb_one = paddle.multiply(dense_inputs, self.dense_w_one)

        # (batch_size, 1)
        y_first_order = paddle.sum(sparse_emb_one, axis=1) + paddle.sum(dense_emb_one, axis=1, keepdim=True)

        # -------------------Field-aware second order term  --------------------
        sparse_embeddings = self.embedding(sparse_inputs_concat)
        # (batch_size, 1, sparse_feature_dim * embedding_size)
        dense_embeddings = paddle.unsqueeze(self.dense_dnn(dense_inputs), axis=1)

        # (batch_size, sparse_num_field, sparse_num_field * embedding_size)
        feat_embeddings = paddle.concat([sparse_embeddings, dense_embeddings], axis=1)

        feat_embeddings = self.cen(feat_embeddings)

        # (batch_size, sparse_num_field, sparse_num_field, embedding_size)
        field_aware_feat_embedding = paddle.reshape(
            feat_embeddings,
            shape=[
                -1, self.sparse_num_field, self.sparse_num_field,
                self.sparse_feature_dim
            ])
        field_aware_interaction_list = []
        for i in range(self.sparse_num_field):
            for j in range(i + 1, self.sparse_num_field):
                # (batch_size, embedding_size)
                tmp = field_aware_feat_embedding[:, i, j, :] * field_aware_feat_embedding[:, j, i, :]
                field_aware_interaction_list.append(tmp if self.is_hadamard_product else paddle.sum(tmp,
                                                                                                    axis=1,
                                                                                                    keepdim=True))
        # 1. dot product, shape: (batch_size, (sparse_field_num * (sparse_field_num - 1)/2))
        # 2. hadamard product, shape: (batch_size, (sparse_field_num * (sparse_field_num - 1)/2) * embedding_size)
        ffm_output = paddle.concat(field_aware_interaction_list, axis=1)
        y_field_aware_second_order = self.mlp(ffm_output)

        return y_first_order, y_field_aware_second_order
