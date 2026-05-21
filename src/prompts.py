import itertools

BASE = 'the {adj} concrete {noun} to {entity}'

def get_vanilla_combination(is_instruct, is_olmo):
    template_ft = 'In a sentence, describe the {adj} concrete {noun} to {entity}.'
    if is_olmo & is_instruct:
        template_ft = '<|im_start|>user\n' + template_ft + '<|im_end|>'
    template_pt = 'If I had to describe the {adj} concrete {noun} to {entity} in a sentence, I would write the following:'
    template = template_ft if is_instruct else template_pt
    adjs = ['best', 'most efficient', 'most promising', 'recommended']
    nouns = ['approach', 'action', 'pathway', 'solution']
    entities = ['survive climate change', 'climate adaptation', 'resolve the climate crisis',
                'achieve long term climate sustainability']
    combinations = itertools.product(adjs, nouns, entities)
    return template, combinations

def get_cot_combination(is_olmo):
    template_instruct3 = 'In a sentence, describe {base}.'
    template_instruct1 = 'Generate a list of 15 ideas on {base} (short title only)'
    instruct2 = '''Go through the list and determine whether the ideas are different and bold, modify the ideas as needed to make them bolder and more different. 
            No two ideas should be the same. This is important!'''
    if is_olmo:
        template_instruct1 = '<|im_start|>user\n' + template_instruct1 + '<|im_end|>'
        instruct2 = '<|im_start|>user\n' + instruct2 + '<|im_end|>'
        template_instruct3 = '<|im_start|>user\n' + template_instruct3 + '<|im_end|>'

    adjs = ['best', 'most efficient', 'most promising', 'recommended']
    nouns = ['approach', 'action', 'pathway', 'solution']
    entities = ['survive climate change', 'climate adaptation', 'resolve the climate crisis',
                'achieve long term climate sustainability']
    combinations = itertools.product(adjs, nouns, entities)
    return [template_instruct1, instruct2, template_instruct3], combinations

def get_combinations(is_cot, is_instruct, is_olmo):
    if is_cot:
        return get_cot_combination(is_olmo)
    else:
        return get_vanilla_combination(is_instruct, is_olmo)

def get_prompt(is_cot, template, adj, noun, entity):
    if not is_cot:
        qa = template.format(adj=adj, noun=noun, entity=entity)
        base = BASE.format(adj=adj, noun=noun, entity=entity)
        return base, qa
    else:
        template_instruct1, instruct2, template_instruct3 = template
        base = BASE.format(adj=adj, noun=noun, entity=entity)
        instruct1 = template_instruct1.format(base=base)
        instruct3 = template_instruct3.format(base=base)
        qas = [instruct1, instruct2, instruct3]
        return base, qas