import subprocess
import jinja2
import gradio
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

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


def create_time_plot(attribution):
    class dataset_config:
      date_interval = 10
      date_max = 800
      date_min = -800

    def bce_ad(d):
      if d < 0:
        return f'{abs(d)} BCE'
      elif d > 0:
        return f'{abs(d)} AD'
      return 0
    #compute scores
    date_pred_y = np.array(attribution.year_scores)
    date_pred_x = np.arange(
      dataset_config.date_min + dataset_config.date_interval / 2,
      dataset_config.date_max + dataset_config.date_interval / 2,
      dataset_config.date_interval)
    date_pred_argmax = date_pred_y.argmax(
    ) * dataset_config.date_interval + dataset_config.date_min + dataset_config.date_interval // 2
    date_pred_avg = np.dot(date_pred_y, date_pred_x)

    # Plot figure
    fig = plt.figure(figsize=(10, 5), dpi=100)

    plt.bar(date_pred_x, date_pred_y, color='#f2c852', width=10., label='Ithaca distribution')
    plt.axvline(x=date_pred_avg, color='#67ac5b', linewidth=2., label='Ithaca average')


    plt.ylabel('Probability', fontsize=14)
    yticks = np.arange(0, 1.1, 0.1)
    yticks_str = list(map(lambda x: f'{int(x*100)}%', yticks))
    plt.yticks(yticks, yticks_str, fontsize=12, rotation=0)
    plt.ylim(0, int((date_pred_y.max()+0.1)*10)/10)

    plt.xlabel('Date', fontsize=14)
    xticks = list(range(dataset_config.date_min, dataset_config.date_max + 1, 25))
    xticks_str = list(map(bce_ad, xticks))
    plt.xticks(xticks, xticks_str, fontsize=12, rotation=0)
    plt.xlim(int(date_pred_avg - 100), int(date_pred_avg + 100))
    plt.legend(loc='upper right', fontsize=12)

    #encode to base64 for html parsing
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = '<div>' + '<img src="data:image/png;charset=utf-8;base64,{}">'.format(encoded) + '</div>'

    return html
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
          prediction_idx=prediction_idx), attrib_dict, create_time_plot(attribution)

with open('example_input.txt', encoding='utf8') as f:
    examples = [line for line in f]
gradio.Interface(
        main,
        inputs=gradio.inputs.Textbox(lines=3),
        outputs=['html', gradio.outputs.Label(label='Geographical Attribution'), 'html'],
        examples=examples,
        title='Spaces Demo for Ithaca',
        description='Restoration and Attribution of ancient Greek texts made by DeepMind. Represent missing characters as "-", and characters to be predicted as "?" (up to 10, does not need to be consecutive)<br> <br><a href="https://ithaca.deepmind.com/" target="_blank">blogpost</a>').launch(enable_queue=True)

