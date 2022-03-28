import subprocess
import jinja2
import gradio

subprocess.run(
        ["curl", "--output", "checkpoint.pkl", "https://storage.googleapis.com/ithaca-resources/models/checkpoint_v1.pkl"])
# Copyright 2021 the Ithaca Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example for running inference. See also colab."""
import functools
import pickle

from ithaca.eval import inference
from ithaca.models.model import Model
from ithaca.util.alphabet import GreekAlphabet
import jax

def load_checkpoint(path):
  """Loads a checkpoint pickle.

  Args:
    path: path to checkpoint pickle

  Returns:
    a model config dictionary (arguments to the model's constructor), a dict of
    dicts containing region mapping information, a GreekAlphabet instance with
    indices and words populated from the checkpoint, a dict of Jax arrays
    `params`, and a `forward` function.
  """

  # Pickled checkpoint dict containing params and various config:
  with open(path, 'rb') as f:
    checkpoint = pickle.load(f)

  # We reconstruct the model using the same arguments as during training, which
  # are saved as a dict in the "model_config" key, and construct a `forward`
  # function of the form required by attribute() and restore().
  params = jax.device_put(checkpoint['params'])
  model = Model(**checkpoint['model_config'])
  forward = functools.partial(model.apply, params)

  # Contains the mapping between region IDs and names:
  region_map = checkpoint['region_map']

  # Use vocabulary mapping from the checkpoint, the rest of the values in the
  # class are fixed and constant e.g. the padding symbol
  alphabet = GreekAlphabet()
  alphabet.idx2word = checkpoint['alphabet']['idx2word']
  alphabet.word2idx = checkpoint['alphabet']['word2idx']

  return checkpoint['model_config'], region_map, alphabet, params, forward


def main(text):

  if not 50 <= len(text) <= 750:
    raise app.UsageError(
        f'Text should be between 50 and 750 chars long, but the input was '
        f'{len(input_text)} characters')

  # Load the checkpoint pickle and extract from it the pieces needed for calling
  # the attribute() and restore() functions:
  (model_config, region_map, alphabet, params,
   forward) = load_checkpoint('checkpoint.pkl')
  vocab_char_size = model_config['vocab_char_size']
  vocab_word_size = model_config['vocab_word_size']

  attribution = inference.attribute(
      text,
      forward=forward,
      params=params,
      alphabet=alphabet,
      region_map=region_map,
      vocab_char_size=vocab_char_size,
      vocab_word_size=vocab_word_size)

  restoration = inference.restore(
      text,
      forward=forward,
      params=params,
      alphabet=alphabet,
      vocab_char_size=vocab_char_size,
      vocab_word_size=vocab_word_size)
  print(attribution.json())
  return attribution.json(), restoration.json()

gradio.Interface(
        main,
        "text",
        ["json", "json"]).launch(enable_queue=True)
