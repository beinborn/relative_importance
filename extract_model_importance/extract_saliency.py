


import numpy as np
import tensorflow as tf
import scipy.special

# The code for calculating sensitivity is based on the TextualHeatmap example by Andreas Madsen: https://colab.research.google.com/github/AndreasMadsen/python-textualheatmap/blob/master/notebooks/huggingface_bert_example.ipynb#scrollTo=X8GJbpoUmYdT
# As described in [Andreas Madsen's distill paper](https://distill.pub/2019/memorization-in-rnns/), the saliency map is computed by measuring the gradient magnitude of the output w.r.t. the input.
def compute_sensitivity(model, embedding_matrix, tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    vocab_size = embedding_matrix.get_shape()[0]
    sensitivity_data = []

    # Iteratively mask each token in the input
    for masked_token_index in range(len(token_ids)):

        if masked_token_index == 0:
            sensitivity_data.append({'token': '[CLR]', 'sensitivity': [1] + [0] * (len(token_ids) - 1)})

        elif masked_token_index == len(token_ids) - 1:
            sensitivity_data.append({ 'token': '[SEP]', 'sensitivity': [0] * (len(token_ids) - 1) + [1]})

        # Get the actual token
        else:
            target_token = tokenizer.convert_ids_to_tokens(token_ids[masked_token_index])

            # integers are not differentable, so use a one-hot encoding of the intput
            token_ids_tensor = tf.constant([token_ids[0:masked_token_index] + [tokenizer.mask_token_id] + token_ids[masked_token_index + 1:]], dtype='int32')
            token_ids_tensor_one_hot = tf.one_hot(token_ids_tensor, vocab_size)
            
            # To select the correct output, create a masking tensor.
            # tf.gather_nd could also be used, but this is easier.
            output_mask = np.zeros((1, len(token_ids), vocab_size))
            output_mask[0, masked_token_index, token_ids[masked_token_index]] = 1
            output_mask_tensor = tf.constant(output_mask, dtype='float32')
            
            # Compute gradient of the logits of the correct target, w.r.t. the input
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(token_ids_tensor_one_hot)
                inputs_embeds = tf.matmul(token_ids_tensor_one_hot,embedding_matrix)
                predict = model({"inputs_embeds": inputs_embeds}).logits
                predict_mask_correct_token = tf.reduce_sum(predict * output_mask_tensor)

            # compute the sensitivity and take l2 norm
            sensitivity_non_normalized = tf.norm(tape.gradient(predict_mask_correct_token, token_ids_tensor_one_hot), axis=2)

            # Normalize by the maximum
            sensitivity_tensor = (sensitivity_non_normalized / tf.reduce_max(sensitivity_non_normalized))
            sensitivity = sensitivity_tensor[0].numpy().tolist()

            sensitivity_data.append({'token': target_token,'sensitivity': sensitivity })

    return sensitivity_data

# We calculate relative saliency by summing the sensitivity a token has with all other tokens
def extract_relative_saliency(model, embeddings,tokenizer, sentence):
    sensitivity_data = compute_sensitivity(model, embeddings, tokenizer, sentence)

    distributed_sensitivity = np.asarray([entry["sensitivity"] for entry in sensitivity_data])
    tokens = [entry["token"] for entry in sensitivity_data]

    # For each token, I sum the sensitivity values it has with all other tokens
    saliency = np.sum(distributed_sensitivity, axis=0)

    # Taking the softmax does not make a difference for calculating correlation
    # It can be useful to scale the salience signal to the same range as the human attention
    relative_saliency = scipy.special.softmax(saliency)
    return tokens, relative_saliency



