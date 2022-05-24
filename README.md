# Automated Dilated Spatio-Temporal Synchronous Graph Modeling for Traffic Prediction

We propose an automated dilated spatio-temporal synchronous graph network, named Auto-DSTSGN for traffic prediction. Specifically, we design an automated dilated spatio-temporal synchronous graph (Auto-DSTSG) module to capture the short-term and long-term spatio-temporal correlations by stacking deeper layers with dilation factors in an increasing order. Further, we propose a graph structure search approach to automatically construct the spatio-temporal synchronous graph that can adapt to different data scenarios. Extensive experiments on four real-world datasets demonstrate that our model can achieve about 10% improvements compared with the state-of-art methods. 

# Requirements
- Python 3.6
- numpy == 1.19.4
- pandas == 1.1.1
- torch >= 1.5

## Usage
Commands for training model in two phases:

- Searching phase:
```bash
python train_multi_step.py  --data /config/individual_3layer_12T.json --runs 5 --epochs 60  --print_every 5 --batch_size 64 --tolerance 15   --node_dim 40   --step_size1 2500 --skip_channels 40 --residual_channels 40  --sts_kernal_size 2 --expid _pems08 --forcp 0 --device cuda:0 --in_dim 1 --max_value 10000
```
- Training phase:
```bash
python train_multi_step_nosearch.py  --data  /config/individual_3layer_12T.json  --runs 5  --epochs 200  --print_every 10 --batch_size 64 --tolerance 30  --node_dim 40  --step_size1 2500 --skip_channels 40 --residual_channels 40  --sts_kernal_size 2 --expid _pems08 --forcp 0 --device cuda:0 --loadid _pems08 --epoch_pretest 10 --LOAD_INITIAL True --in_dim 1 --max_value 10000
```
