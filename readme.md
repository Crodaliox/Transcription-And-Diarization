<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="ressources/UMONS.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Umons Logo"/>
<img src="ressources/insa.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Umons Logo"/>

# Transcription with Diarisation and people Information detection

<em></em>

<!-- BADGES -->
<!-- local repository, no metadata badges. -->

<em>Built with :</em>

<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">


</div>
<br>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)

---

## Overview

This project focuses on speaker diarization and audio transcription using a combination of deep learning models and signal processing techniques. Its goal is to analyze an audio recording, automatically detect and differentiate between speakers, transcribe the spoken content, and associate each part of the transcript with the correct speaker.

This pipeline is especially useful in real-world scenarios such as debates, meetings, or interviews, where it is important not only to transcribe speech but also to understand who is speaking and extract relevant demographic information.

The process begins by segmenting the audio into short clips, extracting speaker embeddings using the ECAPA-TDNN model, and applying clustering algorithms like KMeans or DBSCAN to group similar embeddings. These clusters represent distinct speakers. Additionally, speaker characteristics such as gender and age are estimated using a pretrained classification model: `audeering/wav2vec2-large-robust-6-ft-age-gender`, which enhances the interpretability of the diarization results.

In parallel, the Whisper model from OpenAI is used to transcribe the audio into text with accurate word-level timestamps. These timestamps are matched with the speaker segments to attribute each word or phrase to the corresponding speaker. The final output includes a complete transcription with time-aligned speaker labels, along with metadata such as each speaker’s average age and dominant gender.

The final output includes a `.txt` file containing the full transcription along with time-aligned speaker labels and metadata such as age and dominant gender for each speaker.

Project carried out as part of my internship at the University of Mons.

## Demo

| Example file          | Model whisper                        | Link |
| :-------------------- | :----------------------------------- | :------ |
| **debat2speaker.wav** | Medium              | Radio-télévision belge de la Communauté française (RTBF) : https://www.youtube.com/shorts/NIJ7YveWT6E

| Examples |
| :-------------------- |
|Speaker 0 — Average Age: 46.9, Dominant Gender: male|
|Speaker 1 — Average Age: 66.4, Dominant Gender: male|
|--- Transcription ---|
|[0s - 15s] (Speaker 0) : (0.00s, Locuteur 0)  On  ne  dépense  pas  assez  pour  nos  dépenses  militaires  et  de  défense,  c 'est  une  évidence.  Il  y  a  quelques  heures  encore,  le  nouveau  secrétaire  général  de  l 'OTAN,  Marc  Rutte,  a  déclaré  que  les  2 %  du  PIB  que  nous  devrions  accorder  à  notre  défense,  ce  ne  sera  pas  suffisant.  Et  nous,  nous  ne  sommes  même  pas  à  1 ,3 %  en  Belgique,  nous  sommes  les  dé...|
|[13s - 28s] (Speaker 0) :  Nous  ne  sommes  même  pas  à  1 ,3 %  en  Belgique,  nous  sommes  les  derniers  de  la  classe  derrière  le  Grand -Duché  de  Luxembourg.  Donc,  soyons  honnêtes,  non,  nous  ne  sommes  pas  prêts  aujourd 'hui,  si  vous  me  posez  la  question,  nous  ne  sommes  pas  prêts  à  défendre  nous -mêmes,  notre  défense  européenne  et  l 'Ukraine  sans  les  Américains.  Nous  avons  besoin  aujourd 'hui  des  Américains.|
|[26s - 41s] (Speaker 0, Speaker 1) :  qui  ont  besoin  aujourd 'hui  des  Américains,  c 'est -à -dire  ?  –  Bien  entendu, (29.00s, Locuteur 1)  je  comprends  ce  que  l 'on  dit  à  l 'égard  des  États -Unis,  mais  il  faut  reconnaître  que  les  Européens  se  sont  souvent  protégés  sous  le  parapluie  américain  durant  tant  de  décennies...|
|[39s - 47s] (Speaker 0, Speaker 1) :  temps  de  décennie (40.64s, Locuteur 0)  et  aujourd 'hui  il (42.32s, Locuteur 1)  y  a  à  cet  égard  un  réveil  et  je  pense  que  c 'est  salutaire  pour  le  futur  de  nos  jeunes.|


---

| Component        | Details |
| :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Usage**        | <ul><li>Speaker diarization: automatically detect and separate different speakers in an audio file</li><li>Speech-to-text transcription using Whisper with word-level timestamps</li><li>Speaker metadata extraction (age and gender) via pre-trained classification models</li><li>Automatic export of final transcript to a `.txt` file with aligned speaker labels</li></ul>                                                                                                 |
| **Integrations** | <ul><li>Uses <strong>Whisper</strong> (via faster-whisper) for efficient multilingual transcription</li><li>Uses <strong>SpeechBrain (ECAPA-TDNN)</strong> for speaker embedding extraction</li><li>Integrates <strong>audonnx</strong> model for age and gender classification (e.g., `audeering/wav2vec2-large-robust-6-ft-age-gender`)</li><li>Optional: integrates clustering with <strong>KMeans</strong> or <strong>DBSCAN</strong> to identify unique speakers</li></ul> |
| **Performance**  | <ul><li>Runs on GPU (CUDA-enabled) or CPU, auto-detected</li><li>Fast and memory-efficient processing via `faster-whisper` and segment-based audio splitting</li><li>Includes optional TSNE visualization for embedding clusters</li></ul>                                                                                                                                                                                                                             |
| **Dependencies** | <ul><li>Python libraries: `faster-whisper`, `SpeechBrain`, `audonnx`, `torchaudio`, `soundfile`, `scikit-learn`, `PyYAML`, `matplotlib`</li><li>System dependencies: `ffmpeg` for audio slicing</li><li>Requires <code>torch</code> with GPU support (optional) for best performance</li></ul>                                                                                                                                                                                  |


---

## Getting Started
### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python

- **Requirements:**  
  - `torchaudio`  
  - `faster-whisper`  
  - `speechbrain`  
  - `audonnx`  
  - `audinterface`  
  - `scikit-learn`  
  - `matplotlib`  
  - `soundfile`  
  - `ffmpeg-python`

### Installation

Build DemoStereoscopy from the source and install dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone https://github.com/Crodaliox/...
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd ...
    ```

3. **Install the dependencies:**

    ```sh
    ❯ pip install -r requirements.txt
    ```


## Usage

This script allows you to transcribe a multi-speaker audio file by combining speech recognition, speaker clustering, and optional inference of speaker characteristics (age and gender).

### YAML Configuration File (`transcription_settings.yaml`):

The YAML file contains all the configurable settings for the transcription script. It allows you to customize input paths, model parameters, segmentation settings, clustering, and optional speaker attribute inference without modifying the code.

| Key                    | Description                                                   |
|------------------------|---------------------------------------------------------------|
| `audio_path`           | Path to the audio file to be transcribed. (tested with .wav and .mp3)                    |
| `language`             | Language code for the Whisper model (e.g., `fr`, `en`).       |
| `embedding_dir`        | Temporary directory to store audio segments for embeddings.   |
| `modelWhisp`           | Whisper model to use (e.g., `large-v2`, `base`).              |
| `embedding_duration`   | Duration (in seconds) of audio segments used for speaker embeddings extraction. |
| `transcription_duration` | Duration (in seconds) of audio segments used for transcription. |
| `overlap`              | Overlap duration (in seconds) between transcription segments to avoid cutting words. |
| `affichageTimestampWord` | Whether to display the speaker and timestamp for each word (`true`/`false`). |
| `n_speakers`           | Expected number of speakers (used for KMeans clustering).     |
| `PeopleInformation`    | # Enables or disables estimation of speaker age and gender 

> 💡 If `PeopleInformation` is set to `false`, the script does not use the audonnx (Wav2Vec-style) model for speaker attribute inference, which can save disk space and memory usage.
---

This configuration file makes it easy to adapt the transcription process to different audio files and requirements, controlling everything from segmentation to speaker identification and transcription output format.



### Launch script:

You can launch the script with the following commands on console :

```bash
python TranscriptionAndDiarisation.py

```
The .txt file will be saved on the same directory as the input audio file.

---



<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


---
