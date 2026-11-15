# WorkVL: Work-Environment Research Meets Visual-Language Models

This repository accompanies the paper **“WorkVL: Work-Environment Research Meets Visual-Language Models”**. 11th Workshop and Competition on Affective and Behaviour Analysis in-the-Wild (ABAW), ECCV 2026.

WorkVL investigates whether visual-language models (VLMs) can turn ordinary workplace video into structured, time-stamped descriptions of observable work activities and interactions. These descriptions can then be aggregated into candidate indicators relevant to psychosocial work-environment research.


### Motivation

Psychosocial work-environment research relies predominantly on self-reported data. Such measures are indispensable for experiences, appraisals and strain, but they are less suited to continuously describing observable features of work situations. WorkVL explores a complementary approach: using video to describe activities, interactions and temporal patterns that can be inspected and validated as external observations.

The present case concerns retail CCTV footage. The methodological contribution is broader: it provides a workflow for relating work-environment concepts to observable events, converting video into temporal sequences, and calculating transparent aggregate indicators.

## VL Workflow

The video processing complete the following tasks:

1. Samples short video windows from fixed-camera workplace footage.
2. Prompts a VLM to describe or classify observable activities and interactions.
3. Encodes the resulting descriptions as time-stamped sequences.
4. Identifies recurrent patterns and calculates window-level summary measures.
5. Compares selected outputs with manual annotations.

To run the video model, run the following code from the [video](./video/) folder.

### 1. Create the project environment for the video processing

`conda env create -f environment.yml`
`conda activate video`

### 2. Run the video model

Using the [main.py](./video/main.py) file:

`python -m main.py`

### 3. Identification of patterns

Using the [motifs.py](./video/motifs.py) file:

`python -m motifs.py`

### 4. Additional files

There are sample prompts in [prompts.py](./video/prompts.py) file. Sample outputs for the video model are shown in files [timed_result.csv](./video/timed_result.csv) and [tabular_result.csv](./video/tabular_result.csv).


## Reproducing the analysis

To reproduce the analysis, run the following from the analysis folder.

```bash
# 1. Create the project environment for the an analysis
conda env create -f environment.yml
conda activate workvl

# 2. Place authorised inputs outside version control
#    See docs/data-access.md and docs/annotation-guide.md

# 3. Run the pipeline
python -m src.sample_video
python -m src.run_vlm
python -m src.parse_outputs
python -m src.aggregate_indicators

# 4. Recreate tables and figures
python -m src.make_figures
```

## Citation

If you use this repository, please cite the paper:

```bibtex
@inproceedings{hansen_workvl_2026,
  author    = {Hansen, Claus D. and Campos, David},
  title     = {{WorkVL}: Work-Environment Research Meets Visual-Language Models},
  booktitle = {11th Workshop and Competition on Affective and Behavior Analysis in-the-Wild ({ABAW}), {ECCV}},
  year      = {2026},
  address   = {Malm\"o, Sweden},
}
```


## Contact

Claus D. Hansen  
Department of Society & Politics, Aalborg University  
clausdh@society.aau.dk

