<h1 align="center">Automatic Speech Recognition (ASR)</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
   <a href="#final-results">Final results</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains the end-to-end pipeline for solving ASR task with PyTorch. The model was implemented is [Deep Speech 2](https://arxiv.org/pdf/1512.02595).

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

See [wandb report](https://wandb.ai/free001style/ASR/reports/Report-of-ASR--Vmlldzo5NDc2NDAw) with all experiments.

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment
   using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

   ```bash
   # create env
   conda create -n ASR python=3.10

   # activate env
   conda activate ASR
   ```

1. Install all required packages.

   ```bash
   pip install -r requirements.txt
   ```
2. Download model checkpoint, vocab and language model.

   ```bash
   python download_weights.py
   ```

## How To Use

### Inference

1) If you want only to decode audio to text, your directory with audio should has the following format:
   ```
   NameOfTheDirectoryWithUtterances
   └── audio
        ├── UtteranceID1.wav # may be flac or mp3
        ├── UtteranceID2.wav
        .
        .
        .
        └── UtteranceIDn.wav
   ```
   Run the following command:
   ```bash
   python inference.py datasets=inference_custom inferencer.save_path=SAVE_PATH datasets.test.audio_dir=TEST_DATA/audio
   ```
   where `SAVE_PATH` is a path to save predicted text and `TEST_DATA` is directory with audio.
2) If you have ground truth text and want to evaluate model, make sure that directory with audio and ground truth text has the following format:
   ```
   NameOfTheDirectoryWithUtterances
   ├── audio
   │   ├── UtteranceID1.wav # may be flac or mp3
   │   ├── UtteranceID2.wav
   │   .
   │   .
   │   .
   │   └── UtteranceIDn.wav
   └── transcriptions
       ├── UtteranceID1.txt
       ├── UtteranceID2.txt
       .
       .
       .
       └── UtteranceIDn.txt
   ```
   Then run the following command:
   ```bash
   python inference.py datasets=inference_custom inferencer.save_path=SAVE_PATH datasets.test.audio_dir=TEST_DATA/audio datasets.test.transcription_dir=TEST_DATA/transcriptions
   ```
3) If you only have predicted and ground truth texts and only want to evaluate model, make sure that directory with ones has the following format:
   ```
   NameOfTheDirectoryWithUtterances
    ├── ID1.json # may be flac or mp3
    .
    .
    .
    └── IDn.json

   ID1 = {"pred_text": "ye are newcomers", "text": "YE ARE NEWCOMERS"}
   ```
   Then run the following command:
   ```bash
   python calculate_wer_cer.py --dir_path=DIR
   ```
4) Finally, if you want to reproduce results from [here](#final-results), run the following code:
   ```bash
   python inference.py dataloader.batch_size=500 inferencer.save_path=SAVE_PATH datasets.test.part="test-other"
   ```
   Feel free to choose what kind of metrics you want to evaluate (see [this config](src/configs/metrics/inference.yaml)).

### Training

The model training contains of 3 stages. To reproduce results, train model using the following commands:

1. Train 47 epochs without augmentations

   ```bash
   python train.py writer.run_name="part1" dataloader.batch_size=230 transforms=example_only_instance trainer.early_stop=47
   ```

2. Train 103 epochs with augmentations

   ```bash
   python train.py writer.run_name="part2" dataloader.batch_size=230 trainer.resume_from=part1/model_best.pth datasets.val.part=test-other
   ```

3. Train 15 epochs from new optimizer state

   ```bash
   python train.py -cn=part3 writer.run_name="part3" dataloader.batch_size=230 datasets.val.part=test-other
   ```

   It takes around 57 hours to train model from scratch on A100 GPU.

## Final results

This results were obtained using beam search and language model:

```angular2html
                WER     CER
test-other     18.52   9.62
test-clean     7.47    2.70
```

You can see that using language model yields a very significant quality boost:

```angular2html
             WER(w/o lm)    CER(w/o lm)
test-other      25.35          9.73
```

Finally, beam search also contributes to quality improvement:

```angular2html
             WER(w/o bm)     CER(w/o bm)
test-other      25.80            9.91
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
