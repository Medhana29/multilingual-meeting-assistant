# multilingual-meeting-assistant

A deep-learning case study on near-real-time Indic meeting captioning. A VAD, ASR and machine-translation pipeline turns Telugu and Tamil speech into English captions. LoRA adapters adapt the ASR model to noisy audio, and results come with confidence intervals, entity-preservation checks and streaming-latency measurements.

Telugu and Tamil are evaluated. Hindi appears only in the meeting demo.

---

## 1. Pipeline

```
 meeting audio ──► Silero VAD ──► IndicWhisper ASR ──► IndicTrans2 (indic→en) ──► English captions
 (multi-speaker)   (speech       (Telugu / Tamil /      ai4bharat/indictrans2-      with speaker label
                    segments)     Hindi text)            indic-en-dist-200M          and caption latency
```

The project has four parts:

1. **Baseline assistant and evaluation**: the cascade above, a simulated live meeting, and measurements of ASR, translation, streaming latency and noise robustness.
2. **LoRA fine-tuning for noisy speech**: the baseline's main weakness is heavy noise, so the ASR model is adapted with LoRA adapters, using on-the-fly noise augmentation and SpecAugment.
3. **Fine-tuned system**: the same assistant and evaluation, run again with the merged LoRA models.
4. **Extra metrics**: BERTScore, METEOR, entity preservation (numbers, dates, names) and intent agreement.

---

## 2. Repository layout

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── baseline.ipynb                     # baseline assistant + evaluation (executed)
│   ├── 02-lora-noise-robust.ipynb         # LoRA training + evaluation (executed)
│   ├── finetuned.ipynb                    # same assistant, merged LoRA models (executed)
│   └── extra_metrics_entities_intent.ipynb
├── results/
│   ├── baseline/          # te/, ta/, summary_all_languages.csv
│   ├── lora/              # te/, ta/, summary.csv, training_cost.csv
│   ├── finetuned_system/  # same structure as baseline
│   └── extra_metrics/     # extra_metrics_summary.csv
├── figures/
└── demo/                  # meeting_transcript.csv, meeting_references.csv
```

---

## 3. Models and data

| Component | Source |
|---|---|
| Voice activity detection | Silero VAD |
| ASR | IndicWhisper checkpoints (Whisper-medium fine-tuned for Indic languages), 764 M parameters |
| Translation | `ai4bharat/indictrans2-indic-en-dist-200M` |
| ASR evaluation data | `ekacare/vistaar_small_asr_eval` (FLEURS as fallback) |
| Translation evaluation data | `ai4bharat/IN22-Gen` |
| Extra metrics | `roberta-large` (BERTScore), `dslim/bert-base-NER`, `facebook/bart-large-mnli` |

Model weights and datasets are not included in this repository. Please check each model's and dataset's license before reuse.

---

## 4. How to reproduce (Kaggle)

You need a Kaggle GPU session with Internet on, a Hugging Face token stored as the Kaggle secret `HF_TOKEN` (with access to IndicTrans2), and the IndicWhisper checkpoints attached as datasets (one per language). Run each notebook with **Save & Run All (Commit)** so outputs persist.

| # | Notebook | Needs | Produces |
|---|---|---|---|
| 1 | `baseline.ipynb` | checkpoint datasets, `HF_TOKEN` | `results_multilingual.zip` |
| 2 | `02-lora-noise-robust.ipynb` | checkpoint datasets (independent of step 1) | `lora_results.zip`, merged models |
| 3 | `finetuned.ipynb` | merged models from step 2 | `results_finetuned.zip` |
| 4 | `extra_metrics_entities_intent.ipynb` | output of step 1 | `extra_metrics.zip` |

Run the evaluation with `RUN_LANGS = ["te","ta"]`. The LoRA notebook uses `LANGS = ["te","ta"]`. If `peft` reports an incompatible `torchao` version, remove it with `pip uninstall -y torchao`. The demo meeting is built from public test clips, so no private audio is needed.

---

## 5. Results

All ASR numbers are word error rate (WER, %) unless stated. Lower is better.

### 5.1 Baseline system

| Metric | Telugu | Tamil |
|---|---|---|
| ASR WER / CER (100 clips) | 30.4 / 7.6 | 23.8 / 4.8 |
| Translation, text to English: BLEU / chrF++ (200 sentences) | 40.2 / 63.4 | 32.6 / 57.7 |
| Cascade, speech to English: BLEU / chrF++ (50 sentences) | 62.5 / 76.5 | 62.3 / 76.2 |
| Streaming WER, VAD 500 ms (12 utterances) | 34.3 | 24.8 |
| Streaming WER, fixed 3 s / fixed 5 s chunks | 62.9 / 49.0 | 50.4 / 36.8 |
| Caption latency p50 / p95 (s), VAD 500 ms | 5.5 / 6.9 | 4.1 / 6.8 |
| Captions within 5 s | 35 % | 71 % |

VAD-based segmentation clearly beats fixed-size chunks on accuracy. Differences between the VAD settings (250 to 1200 ms) are small and come from only 12 utterances, so they should not be read as a ranking.

**Noise robustness** (white noise added to 40 clips, WER %):

| SNR | clean | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| Telugu | 30.8 | 33.8 | 43.7 | 50.6 | 63.9 |
| Tamil | 23.2 | 25.6 | 34.1 | 40.8 | 50.7 |

WER roughly doubles at 0 dB. This is the weakness that the LoRA fine-tuning targets.

### 5.2 LoRA fine-tuning

LoRA with r = 32, alpha = 64 and dropout = 0.05 on the attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) of the encoder and decoder. This trains 18.9 M of 782.7 M parameters (about 2.4 %). Training used 800 clips per language, 3 epochs, an effective batch of 8 and a learning rate of 3e-4. It took about 20 to 24 minutes per language with a peak GPU memory of about 4.7 GB.

Test sets: 100 clean clips and 60 clips per noisy condition. Significance is judged from the 95 % bootstrap confidence interval of the difference.

| Language | Condition | Baseline WER | LoRA WER | Change (pp) | Verdict |
|---|---|---|---|---|---|
| Telugu | clean | 37.5 | 36.1 | -1.4 | no clear change |
| Telugu | white noise 0 dB | 70.8 | 67.9 | -2.9 | no clear change |
| Telugu | babble noise 0 dB | 86.7 | 80.1 | -6.6 | **better** |
| Telugu | reverb (unseen) | 57.2 | 55.0 | -2.2 | no clear change |
| Tamil | clean | 25.5 | 25.9 | +0.4 | no clear change |
| Tamil | white noise 0 dB | 52.1 | 54.0 | +2.0 | no clear change |
| Tamil | babble noise 0 dB | 86.0 | 70.9 | -15.1 | **better** |
| Tamil | reverb (unseen) | 41.9 | 43.0 | +1.1 | no clear change |

Telugu at white noise 5 dB also improves significantly (-4.8 pp).

**Summary:** the adapters give a clear gain under heavy babble noise (and at one Telugu white-noise level), do not hurt clean speech, and do not generalise to the unseen reverb condition.

### 5.3 Fine-tuned system compared with the baseline

The full assistant, run again with the merged LoRA models. The test clips and the translation model are the same as in 5.1, and the translation scores are identical.

| Metric | Telugu baseline | Telugu fine-tuned | Tamil baseline | Tamil fine-tuned |
|---|---|---|---|---|
| ASR WER (clean) | 30.4 | 29.6 | 23.8 | 23.6 |
| White-noise WER, 10 dB | 43.7 | 39.1 | 34.1 | 31.4 |
| White-noise WER, 0 dB | 63.9 | 61.8 | 50.7 | 51.2 |
| Cascade BLEU / chrF++ | 62.5 / 76.5 | 56.0 / 73.1 | 62.3 / 76.2 | 66.0 / 77.7 |
| Streaming WER, VAD 500 ms | 34.3 | 30.1 | 24.8 | 136.0 (see below) |
| Caption latency p95 (s) | 6.9 | 6.4 | 6.8 | 23.8 (see below) |

How to read this:

- **ASR and noise:** the fine-tuned models are slightly better on clean and noisy speech in Telugu, and similar in Tamil. These runs use 40 to 100 clips without confidence intervals, so they agree with the larger LoRA test in 5.2 but are not stronger evidence than it.
- **Cascade scores:** these use only 50 sentences, so differences of a few points (in either direction) are within the noise and should not be read as the fine-tuned model hurting or helping translation.
- **Tamil streaming failure:** in the fine-tuned Tamil run, one very short segment (0.4 s, produced by a forced cut) made the model repeat the word "ஆம்" about a hundred times. That repetition loop took 16 s to decode and delayed the segments after it, which caused the 136 % WER (many inserted words) and the 23.8 s p95 latency. The baseline Tamil run did not show this (24.8 %). The other streaming settings for the same run behave normally, for example fixed 5 s chunks reach 33.6 %. This is a genuine limitation of the fine-tuned model in the streaming setting, and it shows that very short segments need to be filtered or guarded.

### 5.4 Extra metrics

BERTScore, METEOR, entity preservation and intent agreement for Telugu and Tamil are in `results/extra_metrics/extra_metrics_summary.csv`.

---

## 6. Limitations

- **Languages:** Telugu and Tamil are evaluated. Hindi appears only in the meeting demo and has no evaluation or fine-tuned model.
- **Demo data:** the demo meeting uses clean benchmark clips, so it shows the pipeline working, not real-world meeting performance. A test on an informal, noisy voice recording produced poor captions, which is consistent with the benchmark data being clean read speech.
- **Noise is synthetic** (added white and babble noise, simulated reverb), not recorded meeting noise.
- **Small samples:** 100 clean and 60 noisy test clips per language for the LoRA test, 40 clips for the noise table in 5.1 and 5.3, 50 sentences for the cascade scores, and 12 utterances in the streaming simulation, with a single training seed. Differences smaller than the confidence intervals should not be read as effects.
- **Streaming robustness:** the fine-tuned Tamil model produced a repetition loop on a 0.4 s segment (see 5.3). A minimum segment length and a repetition guard are not implemented.
- **Possible data overlap:** the base checkpoints were trained on related Indic speech data, and the LoRA train and test clips come from the same benchmark family.
- **Latency:** captions are near-real-time rather than instantaneous. With VAD segmentation only about 5 % of Telugu captions arrive within 3 s, and p95 latency is about 6.4 to 6.9 s.

---

## 7. Credits

IndicWhisper and IndicTrans2 (AI4Bharat), Silero VAD, Hugging Face Transformers and PEFT, and the Vistaar, FLEURS and IN22-Gen datasets.

## 8. Author

Medhana Alwal