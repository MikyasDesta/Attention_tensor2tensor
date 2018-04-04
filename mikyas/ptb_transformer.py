

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import collections

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

# Enable TF Eager execution
from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()

# Other setup
Modes = tf.estimator.ModeKeys

# Setup some directories
data_dir = os.path.expanduser("~/t2t/data")
tmp_dir = os.path.expanduser("~/t2t/tmp")
train_dir = os.path.expanduser("~/t2t/train")
checkpoint_dir = os.path.expanduser("~/t2t/checkpoints")
tf.gfile.MakeDirs(data_dir)
tf.gfile.MakeDirs(tmp_dir)
tf.gfile.MakeDirs(train_dir)
tf.gfile.MakeDirs(checkpoint_dir)
gs_data_dir = "gs://tensor2tensor-data"
gs_ckpt_dir = "gs://tensor2tensor-checkpoints/"




problems.available()




# Fetch the MNIST problem
ptb_problem = problems.problem("languagemodel_ptb10k")
# The generate_data method of a problem will download data and process it into
# a standard format ready for training and evaluation.
ptb_problem.generate_data(data_dir, tmp_dir)



registry.list_models()



#hparams.hidden_size = 64
model_name = "transformer"
hparams_set = "transformer_small"


hparams = trainer_lib.create_hparams(hparams_set, data_dir=data_dir, problem_name="languagemodel_ptb10k")
model = registry.model(model_name)(hparams, Modes.TRAIN)


@tfe.implicit_value_and_gradients
def loss_fn(features):
  _, losses = model(features)
  return losses["training"]

BATCH_SIZE = 32
max_num_of_words = 150
ptb_train_dataset = ptb_problem.dataset(Modes.TRAIN, data_dir)
ptb_train_dataset.output_shapes["targets"] = [max_num_of_words]
ptb_train_dataset = ptb_problem.dataset(Modes.TRAIN, data_dir)
ptb_train_dataset = ptb_train_dataset.repeat(None).padded_batch(BATCH_SIZE, ptb_train_dataset.output_shapes)



optimizer = tf.train.AdamOptimizer()


print(ptb_train_dataset)


NUM_STEPS =10

for count, example in enumerate(tfe.Iterator(ptb_train_dataset)):
  example["targets"] = tf.reshape(example["targets"], [BATCH_SIZE, -1, 1, 1])  # Make it 4D.
  loss, gv = loss_fn(example)
  optimizer.apply_gradients(gv)

  if count % 50 == 0:
    print("Step: %d, Loss: %.3f" % (count, loss.numpy()))
  if count >= NUM_STEPS:
    break

model.set_mode(Modes.EVAL)
ptb_eval_dataset = ptb_problem.dataset(Modes.EVAL, data_dir)

# Create eval metric accumulators for NEG_LOG_PERPLEXITY and APPROX_BLEU

metrics_accum, metrics_result = metrics.create_eager_metrics(
    [metrics.Metrics.NEG_LOG_PERPLEXITY, metrics.Metrics.APPROX_BLEU])

for count, example in enumerate(tfe.Iterator(ptb_eval_dataset)):
  if count >= 200:
    break

  # Make the inputs and targets 4D

  example["targets"] = tf.reshape(example["targets"], [1, -1, 1, 1])

  # Call the model
  predictions, _ = model(example)

  # Compute and accumulate metrics
  metrics_accum(predictions, example["targets"])

# Print out the averaged metric values on the eval data
for name, val in metrics_result().items():
  print("%s: %.2f" % (name, val))

