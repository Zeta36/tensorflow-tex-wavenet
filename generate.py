from __future__ import division
from __future__ import print_function
import math
import argparse
from datetime import datetime
import json
import os

import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, text_reader

SAMPLES = 16000
LOGDIR = './logdir'
WINDOW = 8000
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How many waveform samples to generate')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--window',
        type=int,
        default=WINDOW,
        help='The number of past samples to take into '
        'account at each step')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--text_out_path',
        type=str,
        default=None,
        help='Path to output txt file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=True,
        help='Use fast generation')
    return parser.parse_args()

    
def write_text(waveform, filename):
    text = waveform
    y = []
    for index, item in enumerate(text):
        y.append(chr(text[index]))
    print('Prediction is: ', ''.join(str(e) for e in y))
    y = np.array(y)
    np.savetxt(filename, y.reshape(1, y.shape[0]), delimiter="", newline="\n", fmt="%s")
    print('Updated text file at {}'.format(filename))


def main():
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'])

    samples = tf.placeholder(tf.int32)

    if args.fast_generation:
        next_sample = net.predict_proba_incremental(samples)
    else:
        next_sample = net.predict_proba(samples)

    if args.fast_generation:
        sess.run(tf.initialize_all_variables())
        sess.run(net.init_ops)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.all_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    decode = samples

    quantization_channels = wavenet_params['quantization_channels']
    waveform = [32.]

    last_sample_timestamp = datetime.now()
    for step in range(args.samples):
        if args.fast_generation:
            outputs = [next_sample]
            outputs.extend(net.push_ops)
            window = waveform[-1]
        else:
            if len(waveform) > args.window:
                window = waveform[-args.window:]
            else:
                window = waveform
            outputs = [next_sample]

        # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs, feed_dict={samples: window})[0]
        sample = np.random.choice(
            np.arange(quantization_channels), p=prediction)
        waveform.append(sample)

        # Show progress only once per second.
        current_sample_timestamp = datetime.now()
        time_since_print = current_sample_timestamp - last_sample_timestamp
        if time_since_print.total_seconds() > 1.:
            print('Sample {:3<d}/{:3<d}'.format(step + 1, args.samples),
                  end='\r')
            last_sample_timestamp = current_sample_timestamp

        # If we have partial writing, save the result so far.
        if (args.text_out_path and args.save_every and
                (step + 1) % args.save_every == 0):
            out = sess.run(decode, feed_dict={samples: waveform})
            write_text(out, args.text_out_path)

    # Introduce a newline to clear the carriage return from the progress.
    print()

    # Save the result as a wav file.
    if args.text_out_path:
        out = sess.run(decode, feed_dict={samples: waveform})
        write_text(out, args.text_out_path)

    print('Finished generating.')


if __name__ == '__main__':
    main()
