from transformers import TFBertForMaskedLM, BertTokenizer, TFDistilBertForMaskedLM, DistilBertTokenizer, TFAlbertForMaskedLM, AlbertTokenizer, TFXLNetLMHeadModel, XLNetTokenizer
import tensorflow as tf
import numpy as np

# This is not yet working. We spend many hours on it but haven't figured it out because the documentation of the example code seems to be wrong. 

MODEL_NAME = 'xlnet-large-cased'
model = TFXLNetLMHeadModel.from_pretrained(MODEL_NAME, output_attentions=True)
tokenizer = XLNetTokenizer.from_pretrained(MODEL_NAME)

# We show how to setup inputs to predict a next token using a bi-directional context.

input_ids = tf.constant(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False))[None, :]
print(input_ids)
  # We will predict the masked token

perm_mask = np.zeros((1, input_ids.shape[1], input_ids.shape[1]))

perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token

target_mapping = np.zeros((1, 1, input_ids.shape[1]))  # Shape [1, 1, seq_length] => let's predict one token

target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

#outputs = model(input_ids)
#input_ids: 1, 8, perm_mask: 1,8,8 target_mapping: 1,1,8
#print(input_ids.shape, tf.constant(perm_mask, dtype=tf.float32).shape, tf.constant(target_mapping, dtype=tf.float32).shape)
outputs = model(input_ids,perm_mask=tf.constant(perm_mask, dtype=tf.float32))

next_token_logits = outputs.logits  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

print(next_token_logits[0][-1])
next_token = tokenizer.convert_ids_to_tokens(int(next_token_logits[0][-1]))

print(next_token)

