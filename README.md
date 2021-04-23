This is the code for the ACL submission 2121. 

In order to re-run our analyses, simply run **analyse_all.py.** 
If you want to reproduce the data extraction, run **extract_all.py**. Note that this takes quite long. You need to make sure that you have placed the eye-tracking corpora at extract_human_fixations/data. 

If you want to run the code for another model, you need to modify extract_all.py. 
You can also add another eye-tracking dataset there, but you need to additionally specify a data_extractor. Note that aligning the tokenization of eye-tracking corpora with the language model tokenizers can be tricky. 

**extract_human_fixations**: code to extract the relative fixation duration from two eye-tracking corpora and average it over all subjects. The two corpora are [GECO](https://expsy.ugent.be/downloads/geco/) and [ZUCO](https://osf.io/q3zws/). 

**extract_model_importance**: code to extract saliency-based and attention-based importance from transformer-based language models. 

**analyis**: code to compare and analyze patterns of importance in the human fixation durations and the model data. Also contains code to replicate the plots in the paper. 

**plots**: contains all plots.

**results**: contains intermediate results. 


We use the following packages (see requirements.txt for details): 
numpy, tensorflow, transformers, scikit-learn, spacy, wordfreq, scipy

