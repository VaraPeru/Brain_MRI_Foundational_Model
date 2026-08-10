# BrainIAC: A foundation model for generalized Brain MRI analysis

<p align="center">
  <img src="pngs/brainiac.jpeg" width="800" alt="BrainIAC_V2 Logo"/>
</p>

## Overview

BrainIAC (Brain Imaging Adaptive Core) is vision based foundation model for generalized structural Brain MRI analysis. This repository provides the BrainIAC and downstream model checkpoints, with training/inference pipeline across all downstream tasks. Checkout the [Paper]([https://pmc.ncbi.nlm.nih.gov/articles/PMC11643205/](https://www.nature.com/articles/s41593-026-02202-6))


## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.0+ 



### Setup Environment

```bash
# Clone the repository
git clone https://github.com/YourUsername/BrainIAC_V2.git
cd BrainIAC

# Create conda environment

# Create conda environment
conda create -n brainiac python=3.9
conda activate brainiac
pip install -r requirements.txt
```

## Model Checkpoints

Download the BrainIAC weights and downstream model [checkpoints](https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AG99uZljziHss5zJz4HiFis?rlkey=9w55le6tslwxlfz6c0viylmjb&st=b9cnvwh8&dl=0) and place them in `./src/checkpoints/`:





## Quick Start

See [quickstart.ipynb](./src/quickstart.ipynb) to get started on how to preprocess data, load BrainIAC to extract features, generate and visualize saliency maps. We provide data samples from publicly available [UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/) [License](https://creativecommons.org/licenses/by/4.0/) (with no modifications to the provided preprocessed images) and the [Pixar](https://openneuro.org/datasets/ds000228/versions/1.1.1)  [License](https://creativecommons.org/public-domain/cc0/) dataset in the [sample_data](src/data/sample/processed/) directory. 


## Train and Infer Downstream Models

- [Brain Age Prediction](./docs/downstream_tasks/brain_age_prediction.md)
- [IDH Mutation Classification](./docs/downstream_tasks/idh_mutation_classification.md)
- [Mild Cognitive Impairment Classification](./docs/downstream_tasks/mild_cognitive_impairment_classification.md)
- [Diffuse Glioma Overall Survival Prediction](./docs/downstream_tasks/diffuse_glioma_overall_survival.md)
- [MR Sequence Classification](./docs/downstream_tasks/MR_sequence_classification.md)
- [Time to Stroke Prediction](./docs/downstream_tasks/timetostroke_prediction.md)
- [Tumor Segmentation](./docs/downstream_tasks/tumor_segmentation.md)


## Brainiac Platform

BrainIAC and all the downstream models are hosted at [**Brainiac Platform**](https://www.brainiac-platform.com/) for inference. The platform provides an interface for uploading the structural brain MRI data and running inference on the models including BrainIAC, brain age, MCI classification, time since stroke prediction etc.



## Citation

```bibtex
@article{tak2026generalizable,
  title={A generalizable foundation model for analysis of human brain MRI},
  author={Tak, Divyanshu and Gormosa, B.A. and Zapaishchykova, A. and others},
  journal={Nature Neuroscience},
  year={2026},
  publisher={Springer Nature},
  doi={10.1038/s41593-026-02202-6},
  url={https://doi.org/10.1038/s41593-026-02202-6}
}
```


## License

This project is licensed for non-commercial academic research use only.
Commercial use is not permitted without a separate license from
Mass General Brigham.

For commercial licensing inquiries, please contact the
Mass General Brigham Office of Technology Development. See [LICENSE](LICENSE) for details.


