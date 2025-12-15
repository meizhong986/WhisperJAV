---
license: apache-2.0
---

# Introduction

The MossFormer2_SE_48K model weights for 48 kHz speech enhancement in [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio/tree/main) repo.

This model is trained on large scale datasets inclduing open-sourced and private data.

It enhances speech audios by removing background noise.

# Install

**Clone the Repository**

``` sh
git clone https://github.com/modelscope/ClearerVoice-Studio.git
```

**Create Conda Environment**

``` sh
cd ClearerVoice-Studio
conda create -n clearvoice python=3.8
conda activate clearvoice
pip install -r requirements.txt
```

**Run Script**

Go to `clearvoice/` and use the following examples. The MossFormer2_SE_48K model will be downloaded from huggingface automatically.

Sample example 1: use speech enhancement model `MossFormer2_SE_48K` to process one wave file of `samples/input.wav` and save the output wave file to `samples/output_MossFormer2_SE_48K.wav`

```python
from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

output_wav = myClearVoice(input_path='samples/input.wav', online_write=False)

myClearVoice.write(output_wav, output_path='samples/output_MossFormer2_SE_48K.wav')
```

Sample example 2: use speech enhancement model `MossFormer2_SE_48K` to process all input wave files in `samples/path_to_input_wavs/` and save all output files to `samples/path_to_output_wavs`

```python
from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

myClearVoice(input_path='samples/path_to_input_wavs', online_write=True, output_path='samples/path_to_output_wavs')
```

Sample example 3: use speech enhancement model `MossFormer2_SE_48K` to process wave files listed in `samples/audio_samples.scp' file, and save all output files to 'samples/path_to_output_wavs_scp/'

```python
from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

myClearVoice(input_path='samples/scp/audio_samples.scp', online_write=True, output_path='samples/path_to_output_wavs_scp')
```
