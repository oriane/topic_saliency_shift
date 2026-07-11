def vanilla_generation(model, prompt, n=5, max_new_tokens=100):
    outputs = []
    for i in range(n):
        output = model.generate_text(prompt,
                                     generated_only=True,
                                     generator_args={'max_new_tokens': max_new_tokens,
                                                     'do_sample': True,
                                                     'top_p': 0.9,
                                                     'top_k': 0,
                                                     'temperature': 1},
                                     tokenizer_args={'return_tensors': 'pt'})
        if output:
            output = output[0] if len(output) == 1 else output
        outputs.append(output)
    return outputs

def cot_generation(model, prompt, n=5):
    outputs = []
    for i in range(n):
        output = model.generate_text_cot(prompt,
                                     generated_only=True,
                                     generator_args={'temperature': 1,
                                                     'top_p': 0.9,
                                                     'max_new_tokens': 250},
                                     tokenizer_args={'return_tensors': 'pt'})
        output = output[0] if len(output) == 1 else output
        outputs.append(output)
    return outputs