def to_group_num(quantize_mode):
    if quantize_mode == 'g':
        return 4
    elif quantize_mode.startswith('g'):
        return int(quantize_mode[1:])
    else:
        return -1

def calculate_exp_id(pa=0, pw=0, qa=0, qw=0, apoz=False, sparse_aware=False, naive_p=False, quantize_mode='g', zg=False, zgf=False, structure=False, mask_batchnorm=False, bernoulli=False,
    filter_p=None, unstructure_2nd=False, sp_balance=False, collect_q_stats="", group_both=False, no_layerwise=False, q_then_p=False, layer_pq=False, **kwargs):
    # assert pa == pw
    # assert isinstance(pa, int) and isinstance(pw, int)

    if apoz:
        pa = 'apoz'
        pw = 0

    exp_id = ''

    if qw != 0:
        if qa == qw:
            q_exp_id = f'wa{qa}'
        else:
            q_exp_id = f'w{qw}a{qa}'
    else:
        q_exp_id = ''

    if q_exp_id != '':
        if sparse_aware and not structure:
            q_exp_id += '_spa'
        q_exp_id += ('_' + quantize_mode)

    p_exp_id = ''
    if pa != 0:
        if pw != 0:
            if pa == pw:
                p_exp_id = f'wa{pa}'
            else:
                p_exp_id = f'w{int(pw)}a{pa}'
        else:
            if pa == 'apoz':
                p_exp_id = 'apoz'
            else:
                p_exp_id = f'a{pa}'
    else:
        if pw != 0:
            p_exp_id = f'w{pw}'

    if q_exp_id == '':
        if naive_p:
            exp_id = p_exp_id + '_mag_aistat'
        else:
            exp_id = p_exp_id + '_mag_top_loc'
    else:
        if p_exp_id == '':
            exp_id = q_exp_id
        else:
            exp_id = p_exp_id + '_' + q_exp_id
    
    if zg:
        if zgf:
            exp_id += '_zgf'
        else:
            exp_id += '_zg'
    

    if bernoulli:
        exp_id += '_bnl'
    
    if structure:
        exp_id += '_st'

    if mask_batchnorm:
        exp_id += '_mbn'

    if filter_p is not None:
        exp_id += f'_p{filter_p}'

    if unstructure_2nd:
        exp_id += '_ust2'
        p_exp_id += '_ust2'

    if sp_balance:
        exp_id += '_spb'

    if collect_q_stats:
        exp_id += '_cqs'
        q_exp_id += '_cqs'

    if group_both:
        exp_id += '_gwa'

    if no_layerwise:
        exp_id += '_nly'


    if q_then_p:
        exp_id += '_qp'

    if layer_pq:
        exp_id += '_lpq'

    if bernoulli:
        exp_id += '_bern'

    return exp_id, p_exp_id, q_exp_id


def create_conversions(p_weight_conversions=None, q_weight_conversions=None, p_activation_conversions=None, q_activation_conversions=None, naive_p=False,
                       p_exp_id=None, q_exp_id=None, pw=None, pa=None, apoz=None, structure=False, mask_batchnorm=False, bernoulli=False, filter_p=None, unstructure_2nd=False,
                       sp_balance=False, collect_q_stats="",
                       **kwargs):

    if structure:
        p_weight_conversions = [{**a, "structure": True} for a in p_weight_conversions]
        p_activation_conversions = [{**a, "structure": True} for a in p_activation_conversions]

    if bernoulli and not naive_p:
        for a in p_activation_conversions:
            a["bernoulli"] = True
    
    if filter_p is not None:
        p_activation_conversions = [{**a, "act_filter_p": filter_p} for a in p_activation_conversions]

    if unstructure_2nd:
        p_activation_conversions = [{**a, "preserve_existing_mask": True} for a in p_activation_conversions]    
    
    if sp_balance:
        p_activation_conversions = [{**a, "sp_balance": True} for a in p_activation_conversions]    


    if collect_q_stats:
        q_weight_conversions = [{**a, "collect_q_stats": collect_q_stats} for a in q_weight_conversions]
    
    #     p_activation_conversions = [{**a, "bernoulli": True} for a in p_activation_conversions]
    # if mask_batchnorm:
    #     p_activation_conversions = [{**a, "mask_batchnorm": True} for a in p_activation_conversions]

    return [*(p_weight_conversions if p_exp_id != '' and pw > 0 else []),
            # always here for easier weight reuse
            *(q_weight_conversions if q_exp_id !=
              '' or (pa > 0) else []),
            *(p_activation_conversions if p_exp_id !=
              '' and (pa > 0 or apoz) else []),
            # always here for easier weight reuse
            *(q_activation_conversions if q_exp_id != '' or (pa > 0) else [])]


def switch_quantize_prune(q_exp_id, p_exp_id, quantize_params, prune_params):
    if q_exp_id != '':  
        return quantize_params 
    elif p_exp_id != '' and q_exp_id == '':
        return prune_params
    else:
        raise RuntimeError()