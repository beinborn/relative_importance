#Relative Importance in Sentence Processing

This project contains the code for the following paper: 
Nora Hollenstein, Lisa Beinborn (2021):
Relative Importance for Sentence Processing
*to appear in the proceedings of ACL 2021.*
TODO: add link to paper  

### Re-run experiments
In order to re-run our analyses, simply run **analyse_all.py.** Note that some calculations might take a little. Feel free to comment them out.  

If you want to reproduce the data extraction, run **extract_all.py**. Note that this takes quite long. You need to make sure that you have downloaded the eye-tracking corpora (see extract_human_fixations/README.md for details). 

### Adding models or eye-tracking datasets
If you want to run the code for another model, you need to modify extract_all.py. 
You can also add another eye-tracking dataset there but you would need to implement a new data_extractor. Note that aligning the tokenization of eye-tracking corpora with the language model tokenizers can be tricky. 

### Folder structure
**extract_human_fixations**: code to extract the relative fixation duration from two eye-tracking corpora and average it over all subjects. The two corpora are [GECO](https://expsy.ugent.be/downloads/geco/) and [ZUCO](https://osf.io/q3zws/). 

**extract_model_importance**: code to extract saliency-based and attention-based importance from transformer-based language models. 

**analyis**: code to compare and analyze patterns of importance in the human fixation durations and the model data. Also contains code to replicate the plots in the paper. 

**plots**: contains all plots.

**results**: contains intermediate results. 

### Requirements
We use the following packages (see requirements.txt): 
numpy (1.19.5), tensorflow (2.4.1), transformers (4.2.2), scikit-learn (0.22.2.post1), spacy (2.3.5), wordfreq (2.3.2), scipy (1.4.1)

Note that later versions of transformers might lead to errors. 

To install, create and activate a virtual environment and run: 
pip3 install -r requirements.txt

For the more fine-grained analyses (POS-tags, word frequencies), you need to download the English spaCy model en_core_web_md to your virtual environment: 
python -m spacy download en_core_web_md


