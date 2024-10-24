# Triplet-modality Group-guided Incremental Distillation with Regularized Group Semantic Consistency
[[Project page]](https://github.com/lyy-nlp/MMT_main) 

![Image Description](Figure2.png)
Triplet-modality Group-guided Incremental Distillation with Regularized Group Semantic Consistency for Multi-modal Neural Machine Translation
# Setup

Setup the repository with the following commands:
```bash
conda create MMT_main -f environment.yml
conda activate MMT_main
pip install -r requirements.txt
pip install -e .
```

# Data
txt data we employ the data set [Multi30K dataset](http://www.statmt.org/wmt18/multimodal-task.html),[Flickr30kEn-Jp](https://github.com/nlab-mpg/Flickr30kEnt-JP), [Fashion-MMT](https://github.com/syuqings/Fashion-MMT), [EMMT](https://huggingface.co/datasets/Yaoming95/EMMT). Then use [BPE](https://github.com/rsennrich/subword-nmt) to preprocess the raw data(dataset/data/task1/tok/). Image features are extracted through the pre-trained Resnet-101.  
The data-raw folder above is the data processed by BPE.
The data-bin folder above is the data processed by preprocessed.

##### BPE (learn_joint_bpe_and_vocab.py and apply_bpe.py) 
-s 6000 \
--vocabulary-threshold 1 \

# Step 1: Data preprocess
The data-bin folder above is the data processed by preprocessed.  Then add the pre-trained Resnet-101 image features here and start training.
```bash
sh data-process.sh
```

# Step 2: Training
We need a pre-trained Neural Machine Translation (NMT) model, and we use [Fairseq](https://github.com/facebookresearch/fairseq) as the pre-trained model.
Train a new model on one or across multiple GPUs.
```bash
sh data-train.sh
```

# Step 3: Checkpoints
Loads checkpoints from inputs and returns a model with averaged weights.\
Inputs: An iterable of string paths of checkpoints to load from.\
Returns:A dict of string keys mapping to various values. The 'model' key from the returned dict should correspond to an OrderedDict mapping string parameter names to torch Tensors.
```bash
sh data-checkpoint.sh
```

# Step 4: Generate
Translate pre-processed data with a trained model, evaluate text quality.
```bash
sh data-generate.sh
```


