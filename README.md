# Disentangled Style Representation 

This repository is for the final project for COSI 137B, Information Extraction. In our project, we empirically examine the effectiveness of a previously proposed style transfer system ([John et al., 2018](https://arxiv.org/abs/1808.04339)) on the task of formality encoding.

Note that this project only focuses on *formality encoding*. Therefore, the final paper does not discuss metrics for style transfer such as BLEU. Instead, the transfer performance is discussed in [the presentation of Yonglin's capstone project](https://github.com/yonglin-wang/formality-styler/blob/fe18cc13f261654cb711065c6089cb087feb8163/doc/project_presentation.pdf) (page 9). 

Group members: Yonglin Wang, Xiaoyu Lu

## Relevant Documents

Listed in reverse chronological order...

[Final Written Report](./docs/final-project-report.pdf): Final written report containing the full description of methods and results. 

[Project Proposal](./docs/project-proposal.pdf): Proposed experiment details; our plans have adjusted slightly since. 

[Annotated Bibliography](./docs/annotated-bib.pdf): Preliminary research before landing on our current choice of system.

# Running Our Project

Changes, Procedure, and Results from Yonglin. Note that ours is slightly different from what's described in the original repository. 

## Prerequisites in our experiments

Note that, contrary to the original project, we didn't install KenLM since the associated metrics are outside the scope of our topic. 

* [python=3.6](https://www.python.org/downloads)
* [pip](https://pip.pypa.io/en/stable/installing/)
* [tensorflow==1.12.0](https://pypi.org/project/tensorflow/)
* [numpy==1.17.4](https://pypi.org/project/numpy/)
* [scipy](https://pypi.org/project/scipy/)
* [nltk==3.6.2](https://pypi.org/project/nltk/)
* [spacy==3.0.6](https://pypi.org/project/spacy/)
* [gensim==4.0.1](https://pypi.org/project/gensim/)
* [matplotlib==3.3.4](https://pypi.org/project/matplotlib/)
* [scikit-learn==0.24.2](https://pypi.org/project/scikit-learn/)

## Project Structure and Required Data

To run our current experiments, you will need: 

1. a combined English opionion lexicon ([Direct Download](http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar)), 
2. a 100-dimensional GloVe trained on 6B words (download [here](http://nlp.stanford.edu/data/glove.6B.zip)), and 
3. data splits (matching label and content at each line; you can use string as label, e.g. "informal") from the [GYAFC dataset](https://github.com/raosudha89/GYAFC-corpus). Note that reference files are not needed since the style transfer system in this project utilizes unparallel datasets. 

```
data
├── opinion-lexicon
│  └── sentiment-words.txt
├── glove.6B.100d.txt
├── test.content
├── test.label
├── train.content
├── train.label
├── tune.content
└── tune.label
```



## Commands used in our experiments

### Environment Variables

Before running the commands, declare the following environment variables, note that ```SAVED_MODEL_PATH``` and ```CLASSIFIER_SAVED_MODEL_PATH``` will need to be chagned to the respective time-stamped folder each time a new model is trained. 

```
PROJECT_DIR_PATH=<path to project root>
DATA_DIR=$PROJECT_DIR_PATH/data
TRAINING_TEXT_FILE_PATH=$DATA_DIR/train.content
TRAINING_LABEL_FILE_PATH=$DATA_DIR/train.label
TRAINING_WORD_EMBEDDINGS_PATH=$DATA_DIR/trained_embedding
VALIDATION_TEXT_FILE_PATH=$DATA_DIR/tune.content
VALIDATION_LABEL_FILE_PATH=$DATA_DIR/tune.label
VALIDATION_WORD_EMBEDDINGS_PATH=$DATA_DIR/glove.6B.100d.txt

WORD_EMBEDDINGS_PATH=$TRAINING_WORD_EMBEDDINGS_PATH

CLASS_NUM_EPOCHS=5
NUM_EPOCHS=15
# total vocab after w2v: 124556
VOCAB_SIZE=15000
# DEBUG, INFO, WARNING
LOGGING_LEVEL=INFO

# extract label-correlated words
TEXT_FILE_PATH=$TRAINING_TEXT_FILE_PATH
LABEL_FILE_PATH=$TRAINING_LABEL_FILE_PATH

NUM_SENTENCES=15

# for test, can change later
TEST_TEXT_FILE_PATH=$DATA_DIR/test.content
TEST_LABEL_FILE_PATH=$DATA_DIR/test.label

# main training params TODO change this every time after train validation classifier
SAVED_MODEL_PATH=$PROJECT_DIR_PATH/saved-models/20210504011453
CLASSIFIER_SAVED_MODEL_PATH=$PROJECT_DIR_PATH/saved-models-classifier/20210504010957/
```

### Pretraining


#### Train word embedding model

```bash
./scripts/run_word_vector_training.sh \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--model-file-path ${WORD_EMBEDDINGS_PATH}
```


#### Train validation classifier

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_classifier_training.sh \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--label-file-path ${TRAINING_LABEL_FILE_PATH} \
--training-epochs ${CLASS_NUM_EPOCHS} --vocab-size ${VOCAB_SIZE}
```

This will produce a folder like `saved-models-classifier/xxxxxxxxxx`.

#### Extract label-correlated words

```bash
./scripts/run_word_retriever.sh \
--text-file-path ${TEXT_FILE_PATH} \
--label-file-path ${LABEL_FILE_PATH} \
--logging-level INFO \
--exclude_sentiment
```

The console will out put the top 100 most label-correlated, non-stop, non-sentiment words. 


---


### Style Transfer Model Training


#### Train style transfer model

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_linguistic_style_transfer_model.sh \
--train-model \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--label-file-path ${TRAINING_LABEL_FILE_PATH} \
--training-embeddings-file-path ${TRAINING_WORD_EMBEDDINGS_PATH} \
--validation-text-file-path ${VALIDATION_TEXT_FILE_PATH} \
--validation-label-file-path ${VALIDATION_LABEL_FILE_PATH} \
--validation-embeddings-file-path ${VALIDATION_WORD_EMBEDDINGS_PATH} \
--classifier-saved-model-path ${CLASSIFIER_SAVED_MODEL_PATH} \
--dump-embeddings \
--training-epochs ${NUM_EPOCHS} \
--vocab-size ${VOCAB_SIZE} \
--logging-level="DEBUG"
```

This will produce a folder like `saved-models/xxxxxxxxxx`.
It will also produce `output/xxxxxxxxxx-training` if validation is turned on.


#### Infer style transferred sentences

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_linguistic_style_transfer_model.sh \
--transform-text \
--evaluation-text-file-path ${TEST_TEXT_FILE_PATH} \
--evaluation-label-file-path ${TEST_LABEL_FILE_PATH} \
--saved-model-path ${SAVED_MODEL_PATH} \
--logging-level="DEBUG"
```

This will produce a folder like `output/xxxxxxxxxx-inference`.


#### Generate new sentences

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_linguistic_style_transfer_model.sh \
--generate-novel-text \
--saved-model-path ${SAVED_MODEL_PATH} \
--num-sentences-to-generate ${NUM_SENTENCES} \
--logging-level="DEBUG"
```

This will produce a folder like `output/xxxxxxxxxx-generation`.

---


### Visualizations


#### Plot validation accuracy metrics

```bash
./scripts/run_validation_scores_visualization_generator.sh \
--saved-model-path ${SAVED_MODEL_PATH}
```

This will produce a few files like `${SAVED_MODEL_PATH}/validation_xxxxxxxxxx.svg`


#### Plot T-SNE embedding spaces

```bash
./scripts/run_tsne_visualization_generator.sh \
--saved-model-path ${SAVED_MODEL_PATH}
```

This will produce a few files like `${SAVED_MODEL_PATH}/tsne_plots/tsne_embeddings_plot_xx.svg`


---

## Changes made to the original code

- make all scripts executable: ```chmod -R ./scripts```
- word_retriever.py: 
  - swapped TF tokenizer with nltk tokenizer. Corrected tokenization inconsistency. 
- train_word2vec_model.py: 
  - updated Word2Vec params from ```size=``` to ```vector_size=```
  - Manually defined ```max_vocab_size``` to 15000; anything larger seems to have exhausted the GPU allocated memory. 
- lexicon_helper.py: sklearn no longer offers stopwords; taken off the code. 
- tsne_visualizer.py: line at 27 changed to accommodate dictionary json files that contain either integer key or string key. 
- main.py: if one label is empty during inference, don't error out, print and continue. 

## Word Retriever Results

The following shows the most frequent 100 words in each label, 

1) without filtering out sentiment words:

```
05-04T11:37:21: For label 'informal'
05-04T11:37:21: Most correlated words: ['like', 'dont', 'know', 'think', 'love', 'good', 'want', 'thats', 'cant', 'guys', 'girl', 'tell', 'doesnt', 'time', 'find', 'right', "i'll", 'people', 'gotta', 'best', "can't", 'look', 'yeah', 'song', 'movie', 'girls', 'mean', 'sure', 'maybe', 'thing', 'friends', "he's", "i've", 'life', 'need', 'friend', 'going', 'better', 'gonna', 'kinda', 'pretty', 'great', 'question', 'things', 'said', 'shes', 'music', 'alot', 'talk', 'feel', 'wanna', 'guess', 'answer', 'person', 'little', 'probably', 'nice', 'married', 'stuff', 'women', 'isnt', 'looking', 'watch', 'come', 'work', 'says', 'funny', "that's", 'heard', 'wrong', 'likes', 'hard', 'wait', 'real', 'sorry', 'care', 'hell', 'wont', 'cause', 'help', 'stupid', 'long', 'cool', 'definately', 'depends', 'wants', 'looks', 'songs', 'kids', 'hope', 'yahoo', 'kind', 'play', 'thought', 'hate', 'start', 'dude', 'happy', 'movies', 'whats']
05-04T11:37:21: For label 'formal'
05-04T11:37:21: Most correlated words: ['like', 'know', 'love', 'think', 'good', 'want', 'believe', 'time', 'find', 'song', 'people', 'best', 'enjoy', 'person', 'attractive', 'need', 'women', 'movie', 'relationship', 'tell', 'music', 'answer', 'question', 'friends', 'friend', 'sure', 'feel', 'things', 'great', 'going', 'life', 'woman', 'girl', 'aware', 'information', 'better', 'years', 'look', 'attempt', 'able', 'right', 'mother', 'married', 'said', 'unfaithful', 'understand', 'correct', 'amusing', 'looking', 'similar', 'hope', 'interested', 'favorite', 'nice', 'date', 'work', 'simply', 'homosexual', 'care', 'thing', 'boyfriend', 'watch', 'long', 'prefer', 'heard', 'purchase', 'girls', 'opinion', 'mean', 'enjoyable', 'difficult', 'songs', 'help', 'website', 'received', 'situation', 'luck', 'funny', 'unsure', 'unable', 'wish', 'fond', 'movies', 'agree', 'wrong', 'acceptable', 'sexual', 'large', 'matter', 'talk', 'reason', 'children', 'likely', 'search', 'band', 'appears', 'intercourse', 'choose', 'true', 'television']

```

2) with sentiment words also filtered out:

```
05-04T11:33:39: For label 'formal'
05-04T11:33:39: Most correlated words: ['know', 'think', 'want', 'believe', 'time', 'find', 'song', 'people', 'person', 'need', 'women', 'movie', 'relationship', 'tell', 'music', 'answer', 'question', 'friends', 'friend', 'sure', 'feel', 'things', 'going', 'life', 'woman', 'girl', 'aware', 'information', 'years', 'look', 'attempt', 'able', 'mother', 'married', 'said', 'understand', 'looking', 'similar', 'hope', 'interested', 'date', 'simply', 'homosexual', 'care', 'thing', 'boyfriend', 'watch', 'long', 'heard', 'purchase', 'girls', 'opinion', 'mean', 'songs', 'help', 'website', 'received', 'situation', 'wish', 'movies', 'agree', 'acceptable', 'sexual', 'large', 'matter', 'talk', 'reason', 'children', 'likely', 'search', 'band', 'appears', 'intercourse', 'choose', 'true', 'television', 'feelings', 'girlfriend', 'depends', 'certain', 'desire', 'called', 'young', 'probably', 'play', 'listen', 'require', 'seen', 'different', 'come', 'guys', 'thought', 'individuals', 'leave', 'male', 'possible', 'found', 'year', 'wait', 'wife']
05-04T11:33:39: For label 'informal'
05-04T11:33:39: Most correlated words: ['dont', 'know', 'think', 'want', 'thats', 'cant', 'guys', 'girl', 'tell', 'doesnt', 'time', 'find', "i'll", 'people', 'gotta', "can't", 'look', 'yeah', 'song', 'movie', 'girls', 'mean', 'sure', 'maybe', 'thing', 'friends', "he's", "i've", 'life', 'need', 'friend', 'going', 'gonna', 'kinda', 'question', 'things', 'said', 'shes', 'music', 'alot', 'talk', 'feel', 'wanna', 'guess', 'answer', 'person', 'little', 'probably', 'married', 'stuff', 'women', 'isnt', 'looking', 'watch', 'come', 'says', "that's", 'heard', 'wait', 'real', 'care', 'wont', 'cause', 'help', 'long', 'definately', 'depends', 'wants', 'looks', 'songs', 'kids', 'hope', 'yahoo', 'kind', 'play', 'thought', 'start', 'dude', 'movies', 'whats', 'matter', 'listen', 'called', 'remember', 'wife', 'baby', 'seen', 'sounds', 'woman', 'years', 'thier', 'check', 'actually', 'getting', 'date', 'money', 'talking', 'head', 'read', 'lots']
```



-----

The following is the original repository's README, whose clarity is very much appreciated.

# Linguistic Style-Transfer

Neural network model to disentangle and transfer linguistic style in text

---

## Prerequistites

* [python 3.6](https://www.python.org/downloads)
* [pip](https://pip.pypa.io/en/stable/installing/)
* [tensorflow1.x](https://pypi.org/project/tensorflow/)
* [numpy 1.17.x](https://pypi.org/project/numpy/)
* [scipy](https://pypi.org/project/scipy/)
* [nltk](https://pypi.org/project/nltk/)
* [spacy](https://pypi.org/project/spacy/)
* [gensim](https://pypi.org/project/gensim/)
* [kenlm](https://github.com/kpu/kenlm)
* [matplotlib](https://pypi.org/project/matplotlib/)
* [scikit-learn](https://pypi.org/project/scikit-learn/)

---

## Notes

* Ignore `CUDA_DEVICE_ORDER="PCI_BUS_ID"`, `CUDA_VISIBLE_DEVICES="0"` unless you're training with a GPU
* Input data file format:
    * `${TEXT_FILE_PATH}` should have 1 sentence per line.
    * Similarly, `${LABEL_FILE_PATH}` should have 1 label per line.
* Assuming that you already have [g++](https://gcc.gnu.org/) and [bash](http://tiswww.case.edu/php/chet/bash/bashtop.html) installed, run the following commands to setup the [kenlm](https://github.com/kpu/kenlm) library properly:
    * `wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz`
    * `mkdir kenlm/build`
    * `cd kenlm/build`
    * `sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev` (to install basic dependencies)
    * Install [Boost](https://www.boost.org/):
        * Download boost_1_67_0.tar.bz2 from [here](https://www.boost.org/users/history/version_1_67_0.html)
        * `tar --bzip2 -xf /path/to/boost_1_67_0.tar.bz2`
    * Install [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page):
        * `export EIGEN3_ROOT=$HOME/eigen-eigen-07105f7124f9`
        * `cd $HOME; wget -O - https://bitbucket.org/eigen/eigen/get/3.2.8.tar.bz2 |tar xj`
        * Go back to the `kenlm/build` folder and run `rm CMakeCache.txt`
    * `cmake ..`
    * `make -j2`

---

## Data Sources

### Customer Review Datasets
* Yelp Service Reviews - [Link](https://github.com/shentianxiao/language-style-transfer)
* Amazon Product Reviews - [Link](https://github.com/fuzhenxin/text_style_transfer)

### Word Embeddings
References to `${VALIDATION_WORD_EMBEDDINGS_PATH}` in the instructions below should be replaced by the path to the file `glove.6B.100d.txt`, which can be downloaded from [here](http://nlp.stanford.edu/data/glove.6B.zip).

## Opinion Lexicon
The file `"data/opinion-lexicon/sentiment-words.txt"`, referenced in [global_config.py](https://github.com/vineetjohn/linguistic-style-transfer/blob/master/linguistic_style_transfer_model/config/global_config.py) can be downloaded from below page.
- [Page URL](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html)
- [Direct Download](http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar)

---

## Pretraining


### Train word embedding model
```bash
./scripts/run_word_vector_training.sh \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--model-file-path ${WORD_EMBEDDINGS_PATH}
```


### Train validation classifier

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_classifier_training.sh \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--label-file-path ${TRAINING_LABEL_FILE_PATH} \
--training-epochs ${CLASS_NUM_EPOCHS} --vocab-size ${VOCAB_SIZE}
```

This will produce a folder like `saved-models-classifier/xxxxxxxxxx`.

### Extract label-correlated words
```bash
./scripts/run_word_retriever.sh \
--text-file-path ${TEXT_FILE_PATH} \
--label-file-path ${LABEL_FILE_PATH} \
--logging-level INFO \
--exclude_sentiment
```

The console will out put the top 100 most label-correlated, non-stop, non-sentiment words. 


---


## Style Transfer Model Training


### Train style transfer model

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_linguistic_style_transfer_model.sh \
--train-model \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--label-file-path ${TRAINING_LABEL_FILE_PATH} \
--training-embeddings-file-path ${TRAINING_WORD_EMBEDDINGS_PATH} \
--validation-text-file-path ${VALIDATION_TEXT_FILE_PATH} \
--validation-label-file-path ${VALIDATION_LABEL_FILE_PATH} \
--validation-embeddings-file-path ${VALIDATION_WORD_EMBEDDINGS_PATH} \
--classifier-saved-model-path ${CLASSIFIER_SAVED_MODEL_PATH} \
--dump-embeddings \
--training-epochs ${NUM_EPOCHS} \
--vocab-size ${VOCAB_SIZE} \
--logging-level="DEBUG"
```

This will produce a folder like `saved-models/xxxxxxxxxx`.
It will also produce `output/xxxxxxxxxx-training` if validation is turned on.


### Infer style transferred sentences

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_linguistic_style_transfer_model.sh \
--transform-text \
--evaluation-text-file-path ${TEST_TEXT_FILE_PATH} \
--evaluation-label-file-path ${TEST_LABEL_FILE_PATH} \
--saved-model-path ${SAVED_MODEL_PATH} \
--logging-level="DEBUG"
```

This will produce a folder like `output/xxxxxxxxxx-inference`.


### Generate new sentences

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_linguistic_style_transfer_model.sh \
--generate-novel-text \
--saved-model-path ${SAVED_MODEL_PATH} \
--num-sentences-to-generate ${NUM_SENTENCES} \
--logging-level="DEBUG"
```

This will produce a folder like `output/xxxxxxxxxx-generation`.

---


## Visualizations


### Plot validation accuracy metrics

```bash
./scripts/run_validation_scores_visualization_generator.sh \
--saved-model-path ${SAVED_MODEL_PATH}
```

This will produce a few files like `${SAVED_MODEL_PATH}/validation_xxxxxxxxxx.svg`


### Plot T-SNE embedding spaces

```bash
./scripts/run_tsne_visualization_generator.sh \
--saved-model-path ${SAVED_MODEL_PATH}
```

This will produce a few files like `${SAVED_MODEL_PATH}/tsne_plots/tsne_embeddings_plot_xx.svg`

## Run evaluation metrics


### Style Transfer

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_style_transfer_evaluator.sh \
--classifier-saved-model-path ${CLASSIFIER_SAVED_MODEL_PATH} \
--text-file-path ${GENERATED_TEXT_FILE_PATH} \
--label-index ${GENERATED_TEXT_LABEL}
```

Alternatively, if you have a file with the labels, use the below command instead

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_style_transfer_evaluator.sh \
--classifier-saved-model-path ${CLASSIFIER_SAVED_MODEL_PATH} \
--text-file-path ${GENERATED_TEXT_FILE_PATH} \
--label-file-path ${GENERATED_LABELS_FILE_PATH}
```


### Content Preservation

```bash
./scripts/run_content_preservation_evaluator.sh \
--embeddings-file-path ${VALIDATION_WORD_EMBEDDINGS_PATH} \
--source-file-path ${TEST_TEXT_FILE_PATH} \
--target-file-path ${GENERATED_TEXT_FILE_PATH}
```


### Latent Space Predicted Label Accuracy

```bash
./scripts/run_label_accuracy_prediction.sh \
--gold-labels-file-path ${TEST_LABEL_FILE_PATH} \
--saved-model-path ${SAVED_MODEL_PATH} \
--predictions-file-path ${PREDICTIONS_LABEL_FILE_PATH}
```


### Language Fluency

```bash
./scripts/run_language_fluency_evaluator.sh \
--language-model-path ${LANGUAGE_MODEL_PATH} \
--generated-text-file-path ${GENERATED_TEXT_FILE_PATH}
```

Log-likelihood values are base 10.


### All Evaluation Metrics (works only for the output of this project)

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_all_evaluators.sh \
--embeddings-path ${VALIDATION_WORD_EMBEDDINGS_PATH} \
--language-model-path ${LANGUAGE_MODEL_PATH} \
--classifier-model-path ${CLASSIFIER_SAVED_MODEL_PATH} \
--training-path ${SAVED_MODEL_PATH} \
--inference-path ${GENERATED_SENTENCES_SAVE_PATH}
```
