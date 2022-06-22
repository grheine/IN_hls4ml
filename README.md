# IN_hls4ml

Installation of packages
* Download the environment file deep_tracking_env.yml
```
conda env create -f deep_tracking_env.yml python=3.9
conda activate deep_tracking
pip install -e .
```
* included are also 2 style files for the matplotlib figures

To test the code run the pipeline:
```
git clone https://github.com/grheine/IN_hls4ml.git
cd IN_hls4ml
jupyter notebook IN_pipeline.ipynb
```
A larger data sample of 30.000 events can be downloaded here https://etpwww.etp.kit.edu/~grheine/raw.h5
