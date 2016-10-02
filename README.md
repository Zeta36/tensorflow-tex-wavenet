# A TensorFlow implementation of DeepMind's WaveNet paper for text generation.

This is a TensorFlow implementation of the [WaveNet generative neural
network architecture](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) for <b>text</b> generation.

# Previous work

<table style="border-collapse: collapse">
<tr>
<td>
<p>
Originally, the WaveNet neural network architecture directly generates a raw audio waveform,
showing excellent results in text-to-speech and general audio generation (see the
DeepMind blog post and paper for details).
</p>
<p>
This network models the conditional probability to generate the next
sample in the audio waveform, given all previous samples and possibly
additional parameters.
</p>
<p>
After an audio preprocessing step, the input waveform is quantized to a fixed integer range.
The integer amplitudes are then one-hot encoded to produce a tensor of shape <code>(num_samples, num_channels)</code>.
</p>
<p>
A convolutional layer that only accesses the current and previous inputs then reduces the channel dimension.
</p>
<p>
The core of the network is constructed as a stack of <em>causal dilated layers</em>, each of which is a
dilated convolution (convolution with holes), which only accesses the current and past audio samples.
</p>
<p>
The outputs of all layers are combined and extended back to the original number
of channels by a series of dense postprocessing layers, followed by a softmax
function to transform the outputs into a categorical distribution.
</p>
<p>
The loss function is the cross-entropy between the output for each timestep and the input at the next timestep.
</p>
<p>
In this repository, the network implementation can be found in <a href="./wavenet/model.py">wavenet.py</a>.
</p>
</td>
<td width="300">
<img src="images/network.png" width="300"></img>
</td>
</tr>
</table>

## New approach

This work is based in one implementation of the original WaveNet model ([Wavenet](https://github.com/ibab/tensorflow-wavenet)), but applying some modifications. 

In summary: we are going to use the WaveNet model as a <b>text generator</b>. We'll use raw text data (characters), instead of raw audio files, and once the network is trained, we'll use the conditional probability found to generate samples (characters) into an self-generating process.

Only printable ASCII characters (Dec. 0 up to 255) is supported right now.

## Results

Pretty interesting results are reached!! Feeding the network with enough text and training, the model is able to memorize the probability of the characters disposition (in a lenguage), and generate later even a very similar text!!

For example, using the <a href="./data/texts/ptb.train.txt">Penn Tree Bank</a> (PTB) dataset, and only after 15000 steps of training (with low set of parameters setting) this was the self-generated output (the final loss was around 1.1):

"Prediction is:   300-servenns on the divide  mushin <unk> <unk> attore and <unk> operations losers nis called him for investment it was with  as pursicularly federal and sotheby <unk> d. reported firsts truckhe of the guarantees as paining at the available ransions  i 'm new york for basicane as a facerement of its a set to the u.s. spected on install <unk> <unk> death about in the little there have a $ N million or N N bilot in closing is <unk> <unk> of a trading a congress of society or N cents  for policy half feeling the does n't people of general and the <unk> crafted ended yesterday  still also arjas trading an effectors that a can singaes about N bound who <unk> that mestituty was below for which unrecontimer 's <unk> have day simple d. frisons already earnings on the annual says had minority four-$ N sance for an advised in reclution by <unk> <unk> from $ N million morris selpiculations  the <to will quarter benever on july coming buy-week that the nation tore under new york beyond N million month expected thomas last disappointing to recognition on first had <unk> not year break government  these up why thief east down for his hobses weakness as equiped also plan amr.  him loss appealle they operation after  and the monthly spendings soa $ N million from cansident third-quarter loan was N pressure of new and the intended up he header because in luly of tept. N million crowd up lowers were to passed N while provision according to <unk> and canada said the 1980s defense reporters who west scheduled is a volume at <unk> broke also and <unk> national leader than N years on the sharing N million pro-m was our american piconmentalist profited himses but the measures from N  in N N of social only announcistoner corp. say to average u.j. dey said he crew is vice phick-bar creating the drives will shares of customer with welm reporters involved in the continues after power good operationed retain  medhay  as the end consumer whitecs of the national <unk> inc. closed N million advanc" 

This is really wonderful!! We can see that the original WaveNet model has a great capacity to learn and save long codified text information inside its nodes (and not only audio or image information). This "text generator" WaveNet was able to learn how to write English <b>words and phrases</b> just by predicting characters one by one, and sometimes was able even to learn what word to use based on context.

This output is far to be perfect, but It was trained in a only CPU machine (without GPU) using a low set of parameters configuration in just two hours!! I hope somebody with a better computer can explore the potential of this implementation.

If you want to check this results, you just have to type this in a command line terminal (this will use the trained model checkout I uploaded to the respository):
```bash
python generate.py --text_out_path=output.txt --samples 2000 ./logdir/train/2016-10-02T10-45-10/model.ckpt-14999
```

## Requirements

TensorFlow needs to be installed before running the training script.
TensorFlow 0.10 and the current `master` version are supported.

## Training the network

You can use any text (`.txt`) file.

In order to train the network, execute
```bash
python train.py --data_dir=data
```
to train the network, where `data` is a directory containing `.txt` files.
The script will recursively collect all `.txt` files in the directory.

You can see documentation on each of the training settings by running
```bash
python train.py --help
```

You can find the configuration of the model parameters in [`wavenet_params.json`](./wavenet_params.json).
These need to stay the same between training and generation.

## Generating text

You can use the `generate.py` script to generate audio using a previously trained model.

Run
```
python generate.py --samples 16000 model.ckpt-1000
```
where `model.ckpt-1000` needs to be a previously saved model.
You can find these in the `logdir`.
The `--samples` parameter specifies how many characters samples you would like to generate.

The generated waveform can be stored as a
`.txt` file by using the `--text_out_path` parameter:
```
python generate.py --text_out_path=mytext.txt --samples 1500 model.ckpt-1000
```

Passing `--save_every` in addition to `--text_out_path` will save the in-progress wav file every n samples.
```
python generate.py --text_out_path=mytext.txt --save_every 2000 --samples 1500 model.ckpt-1000
```

Fast generation is enabled by default.
It uses the implementation from the [Fast Wavenet](https://github.com/tomlepaine/fast-wavenet) repository.
You can follow the link for an explanation of how it works.
This reduces the time needed to generate samples to a few minutes.

To disable fast generation:
```
python generate.py --samples 1500 model.ckpt-1000 --fast_generation=false
```

## Missing features

Currently, there is no conditioning on extra information.

# tensorflow-tex-wavenet
