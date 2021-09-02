#@tensorflow_version 1.x
# -*- coding: utf-8 -*-
import modeling as modeling
import tokenization as tokenization
import tensorflow as tf
import json
import collections
import time
import sys
import numpy as np  
import time

tf.reset_default_graph()
label_list = ['0', '1', "2"]

convert_label_to_words = dict()
convert_label_to_words["0"] = "중립"
convert_label_to_words["1"] = "긍정"
convert_label_to_words["2"] = "부정"


class InputFeatures(object):
      def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example

class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label

class PaddingInputExample(object):
      """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

def get_kor_examples(sent):
    examples = []

    for sent__ in sent:
        tokens = tokenizer.tokenize(sent__)
        guid = "%s-%s" % ('test', 0)
        examples.append(InputExample(guid=guid, text=tokens, label='0'))
    return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i
  tokens_a = example.text
  tokens_b = None
  
  if len(tokens_a) > max_seq_length - 2:
    tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)


  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)
    features.append(feature)
  return features


def graph_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      dtype=tf.flags.FLAGS.floatx,
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [num_labels], dtype=tf.flags.FLAGS.floatx, initializer=tf.zeros_initializer())

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.flags.FLAGS.floatx)

    probabilities = tf.nn.softmax(logits, axis=-1)
    predict = tf.argmax(probabilities,axis=-1)
    return predict


def get_feed_dict(features):
    input_ids_list = []
    input_mask_list = []
    segment_ids_list = []
    for feature in features:
        input_ids_list.append(feature.input_ids)
        input_mask_list.append(feature.input_mask)
        segment_ids_list.append(feature.segment_ids)
    return {input_ids: input_ids_list, input_mask:input_mask_list, segment_ids:segment_ids_list}

def get_senti(sentence):
    eval_examples = get_kor_examples(sentence)
    eval_features = convert_examples_to_features(
            examples=eval_examples,
            label_list=label_list,
            tokenizer=tokenizer,
            max_seq_length=512,
    )

    predicts = sess.run(predict_, feed_dict=get_feed_dict(eval_features))
    return predicts

def test_inference(path):
  cnt = 0
  pos = 0
  texts = []
  labels = []
  fr = open(path, 'r', encoding='utf-8-sig')
  lines = fr.readlines()
  batch_size = tf.app.flags.FLAGS.batch_size
  print("total Data : " + str(len(lines) - len(lines)%batch_size))
  print("Inference Start!!...")
  start = time.time()
  for line in lines:
    cnt = cnt + 1
    line = line.split("\t")
    text = line[0].strip("\n").strip()
    label = line[1].strip("\n").strip()
    texts.append(text)
    labels.append(int(label))

    if cnt % batch_size == 0 :
      predict = get_senti(texts)
      labels = np.array(labels)
      pred_sum = np.sum(np.equal(predict, labels))
      pos = pos + pred_sum
      texts = []
      labels = []

  end = time.time() 
  print("pos : " + str(pos) + ", cnt : " + str(cnt - cnt%batch_size))
  print("batch size : ", batch_size)
  print("Accuary : " + str(pos/cnt))
  print("Time : " + str(end - start))

tf.app.flags.DEFINE_string('floatx', 'float32', 'float16 or float32 or mix')
tf.app.flags.DEFINE_integer('batch_size', 1, 'eval batch size')
tf.app.flags.DEFINE_bool('mix', False, 'mix or not mix')


if tf.app.flags.FLAGS.floatx == 'mix':
  tf.app.flags.FLAGS.floatx = 'float32'
  tf.app.flags.FLAGS.mix = True


bert_config = modeling.BertConfig.from_json_file("../model/sa_model/bert_config.json")
tokenizer = tokenization.FullTokenizer(vocab_file="../model/sa_model/vocab.txt", do_lower_case=False)

label_ids = tf.placeholder(shape=[None,None], dtype=tf.int32)
input_ids = tf.placeholder(shape=[None,None], dtype=tf.int32)
input_mask = tf.placeholder(shape=[None,None], dtype=tf.int32)
segment_ids = tf.placeholder(shape=[None,None], dtype=tf.int32)
use_one_hot_embeddings=False

num_labels = 3

predict_ = graph_model(
            bert_config, False, input_ids, input_mask, segment_ids, label_ids, num_labels, use_one_hot_embeddings)
print('\tmodel is initialized!')
tf_config = tf.ConfigProto()

sess = tf.Session(config = tf_config)
sess.run(tf.global_variables_initializer())

model_loader = tf.train.Saver()

if tf.app.flags.FLAGS.mix:
  tf.app.flags.FLAGS.floatx = 'float32'
  model_loader.restore(sess, "../model/fp16_model_8/fp16_model.ckpt")
elif tf.app.flags.FLAGS.floatx == 'float32':
  model_loader.restore(sess, "../model/sa_model/model.ckpt-781")
else :
  model_loader.restore(sess, "../model/hyu_fp16_model/fp16_model.ckpt")
print('\tmodel is set!')


path = "../data/testSet.txt"

test_inference(path)
