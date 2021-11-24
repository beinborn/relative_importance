from transformers import TFBertForMaskedLM, BertTokenizer, TFDistilBertForMaskedLM, DistilBertTokenizer, TFAlbertForMaskedLM, AlbertTokenizer
from extract_model_importance.extract_attention import extract_attention
from extract_model_importance.extract_saliency import extract_relative_saliency
from extract_model_importance import tokenization_util
from extract_human_fixations import data_extractor_geco, data_extractor_zuco

def extract_all_attention(model, tokenizer, sentences, outfile):
    with open(outfile, "w") as attention_file:
        for i, sentence in enumerate(sentences):
            # print(progress
            if i%500 ==0:
                print(i, len(sentences))
            tokens, relative_attention = extract_attention(model, tokenizer, sentence)

        # merge word pieces if necessary
            tokens, merged_attention = tokenization_util.merge_subwords(tokens, relative_attention)
            tokens, merged_attention = tokenization_util.merge_hyphens(tokens, merged_attention)
            attention_file.write(str(tokens) + "\t" + str(merged_attention) + "\n")


def extract_all_saliency(model, embeddings, tokenizer, sentences, outfile):
    with open(outfile, "w") as saliency_file:
        for i, sentence in enumerate(sentences):
            # print(progress
            if i % 500 == 0:
                print(i, len(sentences))
            tokens, saliency = extract_relative_saliency(model, embeddings, tokenizer, sentence)

            # merge word pieces if necessary
            tokens, saliency = tokenization_util.merge_subwords(tokens, saliency)
            tokens, saliency = tokenization_util.merge_hyphens(tokens, saliency)
            saliency_file.write(str(tokens) + "\t" + str(saliency) + "\n")

def extract_all_human_importance(corpus):
    if corpus == "geco":
        data_extractor_geco()
    if corpus == "zuco":
        data_extractor_zuco


corpora_modelpaths = {
    'geco': ['bert-base-uncased', 'distilbert-base-uncased', 'albert-base-v2'],
    'zuco': ['bert-base-uncased', 'distilbert-base-uncased', 'albert-base-v2'],
    'potsdam': [
        'dbmdz/bert-base-german-uncased',
        'distilbert-base-german-cased'
    ],
    'russsent': ['DeepPavlov/rubert-base-cased', 'bert-base-multilingual-cased']
}

for corpus, modelpaths in corpora_modelpaths.items():
    # We skip extraction of human importance here because it takes quite long.
    #extract_all_human_importance(corpus)
    with open("results/" + corpus + "_sentences.txt", "r") as f:
        sentences = f.read().splitlines()
    print("Processing Corpus: " + corpus)

    for mp in modelpaths:
        modelname = mp.split("/")[-1]
        # TODO: this could be moved to a config file
        if modelname.startswith("bert"):
            model = TFBertForMaskedLM.from_pretrained(mp, output_attentions=True)
            tokenizer = BertTokenizer.from_pretrained(mp)
            embeddings = model.bert.embeddings.word_embeddings
        if modelname.startswith('rubert'):
            model = TFBertForMaskedLM.from_pretrained(mp, output_attentions=True, from_pt=True)
            tokenizer = BertTokenizer.from_pretrained(mp)
            embeddings = model.bert.embeddings.word_embeddings
        if modelname.startswith("albert"):
            model = TFAlbertForMaskedLM.from_pretrained(mp, output_attentions=True)
            tokenizer = AlbertTokenizer.from_pretrained(mp)
            embeddings = model.albert.embeddings.word_embeddings
        if modelname.startswith("distil"):
            from_pt = modelname == 'distilbert-base-german-cased'
            model = TFDistilBertForMaskedLM.from_pretrained(mp, output_attentions=True, from_pt=from_pt)
            tokenizer = DistilBertTokenizer.from_pretrained(mp)
            embeddings = model.distilbert.embeddings.word_embeddings

        outfile = "results/" + corpus + "_" + modelname + "_"

        # print("Extracting attention for " + modelname)
        extract_all_attention(model, tokenizer, sentences, outfile+ "attention.txt")

        # Note: Saliency calculation takes much longer than attention calculation.
        print("Extracting saliency for " + modelname)
        extract_all_saliency(model,  embeddings, tokenizer, sentences, outfile + "saliency.txt")
