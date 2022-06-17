# IN_hls4ml

Installation of packages
* Download the environment file deep_tracking_env.yml
```
conda env create -f deep_tracking_env.yml python=3.9
conda activate deep_tracking
pip install -e .
```
To test the code run the pipeline:
```
git clone https://github.com/grheine/IN_hls4ml.git
cd IN_hls4ml
jupyter notebook IN_pipeline.ipynb
```
