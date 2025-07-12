import torch


def text_to_embeds(text, tokenizer, text_encoder, device, 
                   overcome_token_length_limit):
    if overcome_token_length_limit:
        inputs = tokenizer(text, truncation=False, padding='longest', return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        
        encoder_hidden_states = []
        for i in range(0, shape_max_length, tokenizer.model_max_length):
            encoder_hidden_states.append(text_encoder(input_ids[:, i: i + tokenizer.model_max_length])[0])
        encoder_hidden_states = torch.cat(encoder_hidden_states, dim=1)
    else:
        inputs = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = inputs.input_ids
        encoder_hidden_states = text_encoder(input_ids.to(device=device))[0]
    
    return encoder_hidden_states