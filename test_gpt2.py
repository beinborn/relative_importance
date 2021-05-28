# In this file, we have been testing to do the analyses also with GPT-2.
# We are not yet convinced that it works correctly and did not include it to the paper. 


from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import tensorflow as tf
import scipy.special

from extract_model_importance.extract_attention import extract_attention
from extract_model_importance.extract_saliency import extract_relative_saliency
from extract_model_importance import tokenization_util


MODEL_NAME = "gpt2"
model = TFGPT2LMHeadModel.from_pretrained(MODEL_NAME, output_attentions=True)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# Not sure if this is correct. What about the positional embeddings?
embeddings = model.transformer.wte.weight


def compute_sensitivity(model, embedding_matrix, tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    print(text)
    vocab_size = embedding_matrix.get_shape()[0]
    sensitivity_data = []

    # First token, no sensitivity
    sensitivity = [0] * (len(token_ids))
    sensitivity_data.append({'token': tokenizer.convert_ids_to_tokens(token_ids[0]), 'sensitivity': sensitivity})

    # Second token, only first token used for prediction
    if len(token_ids) >1:
        sensitivity = [1.0] + [0] * (len(token_ids) - 1)
        sensitivity_data.append({'token': tokenizer.convert_ids_to_tokens(token_ids[1]), 'sensitivity': sensitivity})

    # Third token. Now, we start.
    # We iterate through all tokens in the sentence, starting with the third one.
    # We feed in all tokens preceding the target token
    # Example: sentence = "A B C D"  target_token = "C" target_token_id = 2, input = "A B", input length = 2
    for target_token_id in range(2, len(token_ids)):
        target_token = tokenizer.convert_ids_to_tokens(token_ids[target_token_id])
        token_ids_tensor = tf.constant(
            [token_ids[0:target_token_id]],
            dtype='int32')

        # This is just for better readability of code in the following lines
        input_length = target_token_id

        # Transform to one-hot vector
        token_ids_tensor_one_hot = tf.one_hot(token_ids_tensor, vocab_size)

        # To select the correct output, create a masking tensor.
        output_mask = np.zeros((1, input_length, vocab_size))


        # We want to check the saliency of each input with respect to the correct prediction
        correct_prediction_id = token_ids[target_token_id]

        # We want to check the sensitivity for the correct prediction at the last input token (target_token_id -1)
        output_mask[0, target_token_id-1, correct_prediction_id] = 1
        output_mask_tensor = tf.constant(output_mask, dtype='float32')

        # Compute gradient of the logits of the correct target, w.r.t. the input
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(token_ids_tensor_one_hot)
            inputs_embeds = tf.matmul(token_ids_tensor_one_hot, embedding_matrix)
            predict = model({"inputs_embeds": inputs_embeds}).logits

            predict_mask_correct_token = tf.reduce_sum(predict * output_mask_tensor)

        # compute the sensitivity and take l2 norm
        sensitivity_non_normalized = tf.norm(tape.gradient(predict_mask_correct_token, token_ids_tensor_one_hot), axis=2)

        # Normalize by the sum (original normalized by max)
        sensitivity_tensor = (sensitivity_non_normalized / tf.reduce_sum(sensitivity_non_normalized))

        # We cannot say anything about the following tokens, so we set sensitivity to 0
        default_following_tokens = [0] * (len(token_ids) - input_length)
        sensitivity = sensitivity_tensor[0].numpy().tolist() + default_following_tokens

        sensitivity_data.append({'token': target_token, 'sensitivity': sensitivity})

    return sensitivity_data


# We calculate relative saliency by summing the sensitivity a token has with all other tokens
def extract_relative_saliency(model, embeddings, tokenizer, sentence):
    sensitivity_data = compute_sensitivity(model, embeddings, tokenizer, sentence)

    print(sensitivity_data)
    distributed_sensitivity = np.asarray([entry["sensitivity"] for entry in sensitivity_data])
    tokens = [entry["token"] for entry in sensitivity_data]
    print(tokens)
    print(distributed_sensitivity)
    # For each token, I sum the sensitivity values it has with all other tokens
    saliency = np.sum(distributed_sensitivity, axis=0)

    # Taking the softmax does not make a difference for calculating correlation
    # It can be useful to scale the salience signal to the same range as the human attention
    relative_saliency = scipy.special.softmax(saliency)
    print(relative_saliency)
    return tokens, relative_saliency



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

# corpora = ["geco", "zuco"]
# for corpus in corpora:
    # with open("results/" + corpus + "_sentences.txt", "r") as f:
    #     sentences = f.read().splitlines()

# Let's start with a few test sentences only
corpus = "test"
sentences = ["how does this work , i wonder", "is that really true"]
print("Processing Corpus: " + corpus)
print("Extracting saliency for " + MODEL_NAME)

outfile = "results/" + corpus + "_" + MODEL_NAME + "_"
extract_all_saliency(model, embeddings, tokenizer, sentences, outfile + "saliency.txt")

