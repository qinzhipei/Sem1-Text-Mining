#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Dict, Optional, Union

import numpy
from overrides import overrides
import torch
from torch import nn
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Elmo, FeedForward, Maxout, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("my_bcn")
class BiattentiveClassificationNetwork(Model): #实现双注意力分类网络，在 AllenNLP 框架内被注册为 "my_bcn"
    """
    This class implements the Biattentive Classification Network model described
    in section 5 of `Learned in Translation: Contextualized Word Vectors (NIPS 2017)
    <https://arxiv.org/abs/1708.00107>`_ for text classification. We assume we're
    given a piece of text, and we predict some output label.
    At a high level, the model starts by embedding the tokens and running them through
    a feed-forward neural net (``pre_encode_feedforward``). Then, we encode these
    representations with a ``Seq2SeqEncoder`` (``encoder``). We run biattention
    on the encoder output representations (self-attention in this case, since
    the two representations that typically go into biattention are identical) and
    get out an attentive vector representation of the text. We combine this text
    representation with the encoder outputs computed earlier, and then run this through
    yet another ``Seq2SeqEncoder`` (the ``integrator``). Lastly, we take the output of the
    integrator and max, min, mean, and self-attention pool to create a final representation,
    which is passed through a maxout network or some feed-forward layers
    to output a classification (``output_layer``).
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    embedding_dropout : ``float``
        The amount of dropout to apply on the embeddings.
    pre_encode_feedforward : ``FeedForward``
        A feedforward network that is run on the embedded tokens before they
        are passed to the encoder.
    encoder : ``Seq2SeqEncoder``
        The encoder to use on the tokens.
    integrator : ``Seq2SeqEncoder``
        The encoder to use when integrating the attentive text encoding
        with the token encodings.
    integrator_dropout : ``float``
        The amount of dropout to apply on integrator output.
    output_layer : ``Union[Maxout, FeedForward]``
        The maxout or feed forward network that takes the final representations and produces
        a classification prediction.
    elmo : ``Elmo``, optional (default=``None``)
        If provided, will be used to concatenate pretrained ELMo representations to
        either the integrator output (``use_integrator_output_elmo``) or the
        input (``use_input_elmo``).
    use_input_elmo : ``bool`` (default=``False``)
        If true, concatenate pretrained ELMo representations to the input vectors.
    use_integrator_output_elmo : ``bool`` (default=``False``)
        If true, concatenate pretrained ELMo representations to the integrator output.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
        BCN 模型是一个用于文本分类的神经网络模型，它的工作流程包括：

词嵌入：

首先，模型将输入文本中的词语通过嵌入层（text_field_embedder）转换为向量表示。这个过程涉及将词语转换为密集的向量。
前馈网络：

然后，这些嵌入向量通过一个前馈神经网络（pre_encode_feedforward）进行初步处理。
编码器：

接着，处理后的向量送入序列到序列编码器（Seq2SeqEncoder，命名为 encoder），以编码整个序列。
双注意力机制：

模型运用双向注意力机制（biattention）处理编码器的输出，得到一个集中关注文本的向量表示。在这种情况下，实际上使用的是自注意力机制（self-attention），因为用于注意力计算的两组表示是相同的。
集成编码器：

然后，将注意力处理后的文本表示与之前的编码器输出结合，再通过另一个序列到序列编码器（Seq2SeqEncoder，命名为 integrator）进行进一步的整合。
池化和最终表示：

最后，从集成编码器得到的输出经过最大值池化（max pooling）、最小值池化（min pooling）、平均池化（mean pooling）和自注意力池化，生成最终的文本表示。
分类输出层：

最终的文本表示通过一个最大化网络（Maxout）或前馈层（FeedForward）来生成分类预测（output_layer）。
参数说明
vocab：必需的参数，用于计算输入/输出投影的大小。
text_field_embedder：用于将输入文本的 tokens 嵌入向量空间。
embedding_dropout：应用于嵌入向量的 dropout 比例。
pre_encode_feedforward：在编码之前对嵌入向量应用的前馈网络。
encoder：用于对 token 进行编码的序列到序列编码器。
integrator：用于整合注意力文本编码和 token 编码的序列到序列编码器。
integrator_dropout：应用于集成编码器输出的 dropout 比例。
output_layer：用于从最终表示生成分类预测的层，可以是最大化网络或前馈网络。
elmo：可选参数，用于将预训练的 ELMo 表示与输入或集成编码器输出连接。
use_input_elmo 和 use_integrator_output_elmo：指定是否将 ELMo 表示分别连接到输入向量和集成编码器输出。
initializer 和 regularizer：用于模型参数的初始化和正则化。
    """
    
    
    '''构造函数
    接受多个参数，包括词汇表（vocab）、文本字段嵌入器（text_field_embedder）、
    多个编码器和前馈网络，以及可选的 Elmo 模型和相关配置。
    '''
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 embedding_dropout: float,
                 pre_encode_feedforward: FeedForward,
                 encoder: Seq2SeqEncoder,
                 integrator: Seq2SeqEncoder,
                 integrator_dropout: float,
                 output_layer: Union[FeedForward, Maxout],
                 elmo: Elmo,
                 use_input_elmo: bool = False,
                 use_integrator_output_elmo: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BiattentiveClassificationNetwork, self).__init__(vocab, regularizer) #调用基类 Model 的构造函数
        
        
        #初始化文本嵌入器、dropout 层、类别数、各种编码器和前馈网络
        self._text_field_embedder = text_field_embedder
        if "elmo" in self._text_field_embedder._token_embedders.keys():  # pylint: disable=protected-access
            raise ConfigurationError("To use ELMo in the BiattentiveClassificationNetwork input, "
                                     "remove elmo from the text_field_embedder and pass an "
                                     "Elmo object to the BiattentiveClassificationNetwork and set the "
                                     "'use_input_elmo' and 'use_integrator_output_elmo' flags accordingly.")
        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self._num_classes = self.vocab.get_vocab_size("labels")

        self._pre_encode_feedforward = pre_encode_feedforward
        self._encoder = encoder
        self._integrator = integrator
        self._integrator_dropout = nn.Dropout(integrator_dropout)

        self._elmo = elmo
        self._use_input_elmo = use_input_elmo
        self._use_integrator_output_elmo = use_integrator_output_elmo
        self._num_elmo_layers = int(self._use_input_elmo) + int(self._use_integrator_output_elmo)
        # Check that, if elmo is None, none of the elmo flags are set.
        if self._elmo is None and self._num_elmo_layers != 0:
            raise ConfigurationError("One of 'use_input_elmo' or 'use_integrator_output_elmo' is True, "
                                     "but no Elmo object was provided upon construction. Pass in an Elmo "
                                     "object to use Elmo.")

        if self._elmo is not None:
            # Check that, if elmo is not None, we use it somewhere.
            if self._num_elmo_layers == 0:
                raise ConfigurationError("Elmo object provided upon construction, but both 'use_input_elmo' "
                                         "and 'use_integrator_output_elmo' are 'False'. Set one of them to "
                                         "'True' to use Elmo, or do not provide an Elmo object upon construction.")
            # Check that the number of flags set is equal to the num_output_representations of the Elmo object
            # pylint: disable=protected-access,too-many-format-args
            if len(self._elmo._scalar_mixes) != self._num_elmo_layers:
                raise ConfigurationError("Elmo object has num_output_representations=%s, but this does not "
                                         "match the number of use_*_elmo flags set to true. use_input_elmo "
                                         "is %s, and use_integrator_output_elmo is %s".format(
                                                 str(len(self._elmo._scalar_mixes)),
                                                 str(self._use_input_elmo),
                                                 str(self._use_integrator_output_elmo)))

        # Calculate combined integrator output dim, taking into account elmo
        if self._use_integrator_output_elmo:
            self._combined_integrator_output_dim = (self._integrator.get_output_dim() +
                                                    self._elmo.get_output_dim())
        else:
            self._combined_integrator_output_dim = self._integrator.get_output_dim()

        self._self_attentive_pooling_projection = nn.Linear(
                self._combined_integrator_output_dim, 1)
        self._output_layer = output_layer

        if self._use_input_elmo:
            check_dimensions_match(text_field_embedder.get_output_dim() +
                                   self._elmo.get_output_dim(),
                                   self._pre_encode_feedforward.get_input_dim(),
                                   "text field embedder output dim + ELMo output dim",
                                   "Pre-encoder feedforward input dim")
        else:
            check_dimensions_match(text_field_embedder.get_output_dim(),
                                   self._pre_encode_feedforward.get_input_dim(),
                                   "text field embedder output dim",
                                   "Pre-encoder feedforward input dim")

        check_dimensions_match(self._pre_encode_feedforward.get_output_dim(),
                               self._encoder.get_input_dim(),
                               "Pre-encoder feedforward output dim",
                               "Encoder input dim")
        check_dimensions_match(self._encoder.get_output_dim() * 3,
                               self._integrator.get_input_dim(),
                               "Encoder output dim * 3",
                               "Integrator input dim")
        if self._use_integrator_output_elmo:
            check_dimensions_match(self._combined_integrator_output_dim * 4,
                                   self._output_layer.get_input_dim(),
                                   "(Integrator output dim + ELMo output dim) * 4",
                                   "Output layer input dim")
        else:
            check_dimensions_match(self._integrator.get_output_dim() * 4,
                                   self._output_layer.get_input_dim(),
                                   "Integrator output dim * 4",
                                   "Output layer input dim")

        check_dimensions_match(self._output_layer.get_output_dim(),
                               self._num_classes,
                               "Output layer output dim",
                               "Number of classes.")

        self.accuracy = CategoricalAccuracy()
        self.f1_measure_positive = F1Measure(
            vocab.get_token_index("positive", "labels"))
        self.f1_measure_negative = F1Measure(
            vocab.get_token_index("negative", "labels"))
        self.f1_measure_neutral = F1Measure(
            vocab.get_token_index("neutral", "labels"))

        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    
    '''它接受一个词汇的字典和一个可选的标签张量，然后返回一个包含模型预测和其他信息的字典'''
    @overrides

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``.
        label : torch.LongTensor, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a
            distribution over the label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        text_mask = util.get_text_field_mask(tokens).float()
        # Pop elmo tokens, since elmo embedder should not be present.
        elmo_tokens = tokens.pop("elmo", None)
        if tokens:
            embedded_text = self._text_field_embedder(tokens)
        else:
            # only using "elmo" for input
            embedded_text = None

        # Add the "elmo" key back to "tokens" if not None, since the tests and the
        # subsequent training epochs rely not being modified during forward()
        if elmo_tokens is not None:
            tokens["elmo"] = elmo_tokens

        # Create ELMo embeddings if applicable
        if self._elmo:
            if elmo_tokens is not None:
                elmo_representations = self._elmo(elmo_tokens)["elmo_representations"]
                # Pop from the end is more performant with list
                if self._use_integrator_output_elmo:
                    integrator_output_elmo = elmo_representations.pop()
                if self._use_input_elmo:
                    input_elmo = elmo_representations.pop()
                assert not elmo_representations
            else:
                raise ConfigurationError(
                        "Model was built to use Elmo, but input text is not tokenized for Elmo.")

        if self._use_input_elmo:
            if embedded_text is not None:
                embedded_text = torch.cat([embedded_text, input_elmo], dim=-1)
            else:
                embedded_text = input_elmo

        dropped_embedded_text = self._embedding_dropout(embedded_text)
        pre_encoded_text = self._pre_encode_feedforward(dropped_embedded_text)
        encoded_tokens = self._encoder(pre_encoded_text, text_mask)

        # Compute biattention. This is a special case since the inputs are the same.
        attention_logits = encoded_tokens.bmm(encoded_tokens.permute(0, 2, 1).contiguous())
        attention_weights = util.masked_softmax(attention_logits, text_mask)
        encoded_text = util.weighted_sum(encoded_tokens, attention_weights)

        # Build the input to the integrator
        integrator_input = torch.cat([encoded_tokens,
                                      encoded_tokens - encoded_text,
                                      encoded_tokens * encoded_text], 2)
        integrated_encodings = self._integrator(integrator_input, text_mask)

        # Concatenate ELMo representations to integrated_encodings if specified
        if self._use_integrator_output_elmo:
            integrated_encodings = torch.cat([integrated_encodings,
                                              integrator_output_elmo], dim=-1)

        # Simple Pooling layers
        max_masked_integrated_encodings = util.replace_masked_values(
                integrated_encodings, text_mask.unsqueeze(2), -1e7)
        max_pool = torch.max(max_masked_integrated_encodings, 1)[0]
        min_masked_integrated_encodings = util.replace_masked_values(
                integrated_encodings, text_mask.unsqueeze(2), +1e7)
        min_pool = torch.min(min_masked_integrated_encodings, 1)[0]
        mean_pool = torch.sum(integrated_encodings, 1) / torch.sum(text_mask, 1, keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self._self_attentive_pooling_projection(
                integrated_encodings).squeeze(2)
        self_weights = util.masked_softmax(self_attentive_logits, text_mask)
        self_attentive_pool = util.weighted_sum(integrated_encodings, self_weights)

        pooled_representations = torch.cat([max_pool, min_pool, mean_pool, self_attentive_pool], 1)
        pooled_representations_dropped = self._integrator_dropout(pooled_representations)

        logits = self._output_layer(pooled_representations_dropped)
        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {'logits': logits, 'class_probabilities': class_probabilities}
        if label is not None:
            loss = self.loss(logits, label)
            self.accuracy(logits, label)
            self.f1_measure_positive(logits, label)
            self.f1_measure_negative(logits, label)
            self.f1_measure_neutral(logits, label)
            output_dict["loss"] = loss

        return output_dict



    '''解码函数：用于将模型的输出（例如类别概率）转换为更易理解的形式（例如标签名称）'''
    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["class_probabilities"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict
    
    '''计算和返回模型的性能指标，如准确率和 F1 分数'''
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        _, recall_positive, f1_measure_positive = self.f1_measure_positive.get_metric(
            reset)
        _, recall_negative, f1_measure_negative = self.f1_measure_negative.get_metric(
            reset)
        _, recall_neutral, _ = self.f1_measure_neutral.get_metric(reset)
        accuracy = self.accuracy.get_metric(reset)

        avg_recall = (recall_positive + recall_negative + recall_neutral) / 3.0
        macro_avg_f1_measure = (
            f1_measure_positive + f1_measure_negative) / 2.0

        results = {
            'accuracy': accuracy,
            'avg_recall': avg_recall,
            'f1_measure': macro_avg_f1_measure
        }
        return results
    
    
    
    '''从参数创建模型的类方法'''
    # The FeedForward vs Maxout logic here requires a custom from_params.
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BiattentiveClassificationNetwork':  # type: ignore
        # pylint: disable=arguments-differ
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab=vocab, params=embedder_params)
        embedding_dropout = params.pop("embedding_dropout")
        pre_encode_feedforward = FeedForward.from_params(params.pop("pre_encode_feedforward"))
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        integrator = Seq2SeqEncoder.from_params(params.pop("integrator"))
        integrator_dropout = params.pop("integrator_dropout")

        output_layer_params = params.pop("output_layer")
        if "activations" in output_layer_params:
            output_layer = FeedForward.from_params(output_layer_params)
        else:
            output_layer = Maxout.from_params(output_layer_params)

        elmo = params.pop("elmo", None)
        if elmo is not None:
            elmo = Elmo.from_params(elmo)
        use_input_elmo = params.pop_bool("use_input_elmo", False)
        use_integrator_output_elmo = params.pop_bool("use_integrator_output_elmo", False)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   embedding_dropout=embedding_dropout,
                   pre_encode_feedforward=pre_encode_feedforward,
                   encoder=encoder,
                   integrator=integrator,
                   integrator_dropout=integrator_dropout,
                   output_layer=output_layer,
                   elmo=elmo,
                   use_input_elmo=use_input_elmo,
                   use_integrator_output_elmo=use_integrator_output_elmo,
                   initializer=initializer,
                   regularizer=regularizer)