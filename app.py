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

def get_subregion_name(id, region_map):
  return region_map['sub']['names_inv'][region_map['sub']['ids_inv'][id]]

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


  restore_template = jinja2.Template("""<!DOCTYPE html>
    <html>
    <head>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400&family=Roboto:wght@400&display=swap" rel="stylesheet">
    <style>
    body {
      font-family: 'Roboto Mono', monospace;
      font-weight: 400;
    }
    .container {
      overflow-x: scroll;
      scroll-behavior: smooth;
    }
    table {
      table-layout: fixed;
      font-size: 16px;
      padding: 0;
      white-space: nowrap;
    }
    table tr:first-child {
      font-weight: bold;
    }
    table td {
      border-bottom: 1px solid #ccc;
      padding: 3px 0;
    }
    table td.header {
      font-family: Roboto, Helvetica, sans-serif;
      text-align: right;
      position: -webkit-sticky;
      position: sticky;
      background-color: white;
    }
    .header-1 {
      background-color: white;
      width: 120px;
      min-width: 120px;
      max-width: 120px;
      left: 0;
    }
    .header-2 {
      left: 120px;
      width: 50px;
      max-width: 50px;
      min-width: 50px;
      padding-right: 5px;
    }
    table td:not(.header) {
      border-left: 1px solid black;
      padding-left: 5px;
    }
    .header-2col {
      width: 170px;
      min-width: 170px;
      max-width: 170px;
      left: 0;
      padding-right: 5px;
    }
    .pred {
      background: #ddd;
    }
    </style>
    </head>
    <body>
    <div class="container">
    <table cellspacing="0">
      <tr>
        <td colspan="2" class="header header-2col">Input text:</td>
        <td>
        {% for char in restoration_results.input_text -%}
          {%- if loop.index0 in prediction_idx -%}
            <span class="pred">{{char}}</span>
          {%- else -%}
            {{char}}
          {%- endif -%}
        {%- endfor %}
        </td>
      </tr>
      <!-- Predictions: -->
      {% for pred in restoration_results.predictions[:3] %}
      <tr>
        <td class="header header-1">Hypothesis {{ loop.index }}:</td>
        <td class="header header-2">{{ "%.1f%%"|format(100 * pred.score) }}</td>
        <td>
          {% for char in pred.text -%}
            {%- if loop.index0 in prediction_idx -%}
              <span class="pred">{{char}}</span>
            {%- else -%}
              {{char}}
            {%- endif -%}
          {%- endfor %}
        </td>
      </tr>
      {% endfor %}
    </table>
    </div>
    <script>
    document.querySelector('#btn').addEventListener('click', () => {
      const pred = document.querySelector(".pred");
      pred.scrollIntoViewIfNeeded();
    });
    </script>
    </body>
    </html>
    """)
  locations = []

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

  prediction_idx = set(i for i, c in enumerate(restoration.input_text) if c == '?')

  attrib_dict = {get_subregion_name(l.location_id, region_map): l.score for l in attribution.locations[:3]}
  return restore_template.render(
          restoration_results=restoration,
          prediction_idx=prediction_idx), attrib_dict
examples = [
    'ed??e? t?? ß????? ?a? t?? d?µ?? ??s?st?at?? e?pe- epe?d? d??fa??? a??? a?a??? ?? d?ate?e? pe?? d?????? ded???a? t?? ----- d??fa??? ?a???-------- --??a??? p???e??? e??a? d--------- a?t?? ?a? e??????? ?-- e??a? a?t??? ate?e?a? e? d???? pa?t?? ?a? ??? ?a? ????a? e??t?s?? ?a? p??s?d?? p??? t?µ ß????? ?a? t?? d?µ?? p??t??? µeta ta ?e?a ?a? ta a??a ?sa ?a? t??? a????? p???e???? ?a? e?e??eta?? t?? ?e??? ded?ta? pa?a ---??? a?a??a?a? de t?de ?????????a t?? ß????? e?? -----------???? t??? -e -----------------------.',
    '?e?? ep? ????f?µ? a????t??. s?µµa??a a???a??? ?a? ?etta??? e?? t?? ae? ??????. ed??e? t?? ?????? ?a? t?? d?µ??. ????t?? ep??ta?e?e? ?a????? ?a???a??? fa???e?? e??aµµate?e? a???pp?? aµf?t??p??e? epestate? d?de?ate? t?? p??ta?e?a?. e-??est?d?? e?pe? -e-- ?? ?e???s?? ?? p-esße?? t?? ?etta??- e??f?s?a- t?? d-µ?? de?es?a? t?? s?µµa??a? t??-? a?a??? ?-?a ep-??e????ta- ?? ?etta??- e??a? de a?-?-? t?- s?µµ-??a? p??? a???a??? e?? -?? a?e? ??????. e?-a? de ?a? t??? a???a??? s?µµ-?-? apa?ta? ?etta??- s?µµ-??? ?a? t?? -etta??? a--?a???.',
    '????. ed??e? t?? ß????? ?a? t?? d?µ??. µe?d??e?? e?pe? ????? ep?state epe?d? ep??t?t?? a??? f???t?µ?? est?? pe?? t?? p???? t?? a??es??e?? ?a? t?? af????µe??? e?? ???a? p??e? a?a??? ?t? a? d???ta? ?a? ????? ??? e???? ded???a? t?? d?µ?? e??a? ep??t?t?? ?a??????t?? ???a??? p???e??? t?? p??e?? t?? a??es??e?? µeta t?? ?pa????t?? ?a? e??a? a?t?? p??s?d?? p??? te t?? d?µ?? ?a? t?? ß????? p??t?? µeta ta ?e?a a?a??a?a? de a?t?? t?? p???e---? e?? t? ??a??? ep?µe?????a? -- f??t---.'
    ]

gradio.Interface(
        main,
        inputs="text",
        outputs=["html", gradio.outputs.Label(label="Geographical Attribution")],
        examples=examples).launch(enable_queue=True)
