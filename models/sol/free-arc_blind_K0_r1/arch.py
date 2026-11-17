import torch
import math


def add_phrases(w, b, sequence, width):
    code = torch.zeros_like(sequence[:, :sequence.shape[1] - width + 1])
    valid = torch.ones_like(code, dtype=torch.bool)
    for offset in range(width):
        part = sequence[:, offset:offset + code.shape[1]]
        code = code * w['base'] + part
        valid = valid * (part > 0) * (part < w['base'])
    location = torch.searchsorted(w['phrase_codes'], code.contiguous())
    location = location.clamp_max(w['phrase_codes'].shape[0] - 1)
    valid = valid * (w['phrase_codes'][location] == code)
    phrase = w['phrase_ids'][location] * valid.to(torch.int64)
    b.scatter_add_(1, phrase, torch.ones_like(phrase, dtype=torch.float32))
    return b


def bag(w, tokens):
    ids = w['lex'][tokens]
    b = torch.zeros((tokens.shape[0], w['idf'].shape[0]), device=tokens.device)
    b.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.float32))
    single = w['single_feature'][ids]
    b.scatter_add_(1, single, torch.ones_like(single, dtype=torch.float32))
    # Remove known grammatical words, but retain unknown words as barriers.
    # This prevents fabricating a phrase across an unrecognized content word.
    position = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    content = (ids != 0) + (w['skip'][tokens] == 0)
    order = torch.where(content > 0, position, position + tokens.shape[1]).argsort(-1)
    barrier_ids = torch.where((ids == 0) * (w['skip'][tokens] == 0), w['base'], ids)
    sequence = barrier_ids.gather(1, order)
    b = add_phrases(w, b, sequence, 2)
    b = add_phrases(w, b, sequence, 3)
    b = add_phrases(w, b, sequence, 4)
    b[:, 0] = 0
    b = b.clamp_max(1)
    b[:, w['override_dst']] = b[:, w['override_dst']] * (1 - b[:, w['override_src']])
    states = b[:, w['quantity_state']]
    ambiguous = (states.sum(-1, keepdim=True) > 1).expand_as(states)
    suppress = torch.zeros_like(b)
    indices = w['quantity_state'].reshape(1, -1).expand(tokens.shape[0], -1)
    suppress.scatter_add_(1, indices, ambiguous.reshape(tokens.shape[0], -1).to(torch.float32))
    return b * (suppress == 0)


def affinity(overlap, mass):
    coverage = (overlap / mass).clamp_max(1)
    return overlap / torch.sqrt(mass + 2) * (0.35 + 0.65 * coverage)


def question_focus(w, q):
    position = torch.arange(q.shape[1], device=q.device).unsqueeze(0)
    priority = w['ask'][q]
    # Prefer the final which/what request. Subordinate why/when clauses cannot
    # replace it; absent a selection request, prefer the first other wh word.
    score = (priority == 2) * (3 * q.shape[1] + position)
    score = score + (priority == 1) * (q.shape[1] - position)
    start = score.argmax(-1, keepdim=True)
    mask = position >= start
    return q * mask.to(torch.int64)


def request_negation(w, focused):
    boundary_count = w['scope_boundary'][focused].cumsum(-1)
    direct_not = w['not_word'][focused] * (boundary_count == 0)
    # Nested negative conditions after 'shows', 'when', 'that', etc. describe
    # a situation rather than asking for an unsupported answer.
    count = (w['neg'][focused] + direct_not).sum(-1)
    return count.remainder(2)


def comparison_polarity(w, focused):
    # 'Least' is already represented by the descending ordering cue. A direct
    # 'not' or 'except' reverses a comparison; 'least' alone must not do so twice.
    negative = request_negation(w, focused)
    negative = (negative - w['least'][focused].sum(-1)).remainder(2)
    return 1 - 2 * negative


def physical_units(w, tokens):
    raw = w['units'][tokens]
    cubic = w['cubic'][tokens]
    before = cubic.roll(1, dims=-1)
    after = cubic.roll(-1, dims=-1)
    before[..., 0] = 0
    after[..., -1] = 0
    cubed = ((before + after) > 0).to(torch.float32)
    length = raw[..., 2]
    volume = raw[..., 1] + length.pow(3) * 1000000 * cubed
    distance = length * (1 - cubed)
    return torch.cat((raw[..., :1], volume.unsqueeze(-1), distance.unsqueeze(-1), raw[..., 3:]), dim=-1)


def measurements(w, q):
    numbers = w['number'][q]
    units = physical_units(w, q)
    position = torch.arange(q.shape[1], device=q.device).unsqueeze(0)
    preceding = torch.full_like(numbers, -1)
    # Only grammatical 'of' and a cubic marker may intervene between a number
    # and its unit. A previous unit can never donate its number to a later unit.
    bridge = w['unit_bridge'][q] > 0
    for gap in range(1, 4):
        candidate = numbers.roll(gap, dims=1)
        intervening = torch.ones_like(bridge)
        for inner in range(gap - 1):
            intervening = intervening * bridge.roll(inner + 1, dims=1)
        valid = (preceding < 0) * (candidate >= 0) * (position >= gap) * intervening
        preceding = torch.where(valid, candidate, preceding)
    next_unit = ((units.sum(-1) > 0) * (w['skip'][q] == 0)).roll(-1, dims=1)
    next_per = w['per_word'][q].roll(-1, dims=1) > 0
    next_unit[:, -1] = 0
    next_per[:, -1] = 0
    simple = ((next_unit + next_per) == 0).unsqueeze(-1)
    valid_unit = (units > 0) * (preceding >= 0).unsqueeze(-1) * simple
    counts = valid_unit.sum(1)
    readings = (units * preceding.clamp_min(0).unsqueeze(-1) * valid_unit).sum(1)
    values = readings / counts.clamp_min(1)
    values = torch.cat((values, torch.ones((q.shape[0], 1), device=q.device)), dim=-1)
    known = torch.cat(((counts == 1), torch.ones((q.shape[0], 1), device=q.device, dtype=torch.bool)), dim=-1)
    ambiguous = ((numbers[:, :-1] >= 0) * (numbers[:, 1:] >= 0)).sum(-1) > 0
    return values, known, ambiguous


def option_measurement(w, o):
    units = physical_units(w, o)
    numbers = w['number'][o]
    previous_number = (numbers >= 0).roll(1, dims=-1)
    previous_per = (w['per_word'][o] > 0).roll(1, dims=-1)
    previous_unit = ((units.sum(-1) > 0) * (w['skip'][o] == 0)).roll(1, dims=-1)
    previous_number[..., 0] = 0
    previous_per[..., 0] = 0
    previous_unit[..., 0] = 0
    # The lowercased ampere symbol 'a' must not turn an ordinary article into
    # a current unit. Recognize it only in an actual numerical/unit context.
    unit_context = (w['skip'][o] == 0) + previous_number + previous_per + previous_unit
    units = units * (unit_context > 0).unsqueeze(-1)
    present = (units > 0).sum(-2) > 0
    scale = units.amax(-2)
    scale = torch.where(present, scale, torch.ones_like(scale))
    positions = torch.arange(o.shape[-1], device=o.device).reshape(1, 1, -1, 1)
    first = torch.where(units > 0, positions, o.shape[-1]).amin(-2)
    powers = w['math_unit_power']
    required = powers != 0
    disagreement = (present.unsqueeze(2).unsqueeze(3) != required).sum(-1)
    matches = (disagreement == 0) * w['math_unit_valid']
    numerator_last = torch.where(powers > 0, first.unsqueeze(2).unsqueeze(3), -1).amax(-1)
    denominator_first = torch.where(powers < 0, first.unsqueeze(2).unsqueeze(3), o.shape[-1]).amin(-1)
    matches = matches * (numerator_last < denominator_first)
    unitless = present.sum(-1) == 0
    first_pattern = torch.arange(powers.shape[1], device=o.device) == 0
    matches = matches + unitless.unsqueeze(-1).unsqueeze(-1) * first_pattern
    conversion = scale.unsqueeze(2).unsqueeze(3).pow(powers).prod(-1)
    conversion = (conversion * matches).sum(-1) / matches.sum(-1).clamp_min(1)
    compatible = matches.sum(-1) > 0
    return conversion, compatible


def arithmetic(w, q, o, focused_bag):
    values, known, ambiguous = measurements(w, q)
    lhs = values[:, w['math_left']]
    rhs = values[:, w['math_right']]
    prediction = lhs * rhs.clamp_min(0.000001).pow(w['math_op'])
    active = known[:, w['math_left']] * known[:, w['math_right']]
    active = active * ((rhs > 0) + (w['math_op'] > 0) > 0)
    active = active * (focused_bag @ w['math_trigger'].t() > 0) * (~ambiguous).unsqueeze(1)
    option_numbers = w['number'][o]
    numeric_count = (option_numbers >= 0).sum(-1)
    first = torch.where(option_numbers >= 0, torch.arange(o.shape[-1], device=o.device), o.shape[-1]).argmin(-1, keepdim=True)
    numeric_value = option_numbers.gather(-1, first).squeeze(-1).clamp_min(0)
    conversion, dimensional_match = option_measurement(w, o)
    observed = numeric_value.unsqueeze(-1) * conversion
    error = (observed - prediction.unsqueeze(1)).abs() / (prediction.abs().unsqueeze(1) + 1)
    compatible = (numeric_count == 1).to(torch.float32)
    enough = (compatible.sum(1) >= 3).to(torch.float32)
    contribution = torch.exp(-w['knobs'][9] * error) * active.unsqueeze(1) * dimensional_match
    return w['knobs'][8] * contribution.sum(-1) * compatible * enough.unsqueeze(1)


def ordered_properties(w, qb, ob):
    overlap = qb @ w['rank_trigger'].t()
    size = w['rank_trigger'].sum(-1).clamp_min(1)
    active = (overlap / size >= 0.74).to(torch.float32)
    up = (qb @ w['rank_up'].t() > 0).to(torch.float32)
    down = (qb @ w['rank_down'].t() > 0).to(torch.float32)
    direction = up - down
    matches = ob[:, :, w['rank_entity']] * (w['rank_entity'] != 0)
    priority = matches * w['rank_specificity']
    selected = (priority == priority.amax(-1, keepdim=True)) * (priority > 0)
    rank_values = (selected * w['rank_number']).sum(-1) / selected.sum(-1).clamp_min(1)
    known = selected.sum(-1) > 0
    enough = (known.sum(1) >= 2).to(torch.float32)
    ranking = rank_values * known * (active * direction * enough).unsqueeze(1)
    return w['knobs'][5] * ranking.sum(-1)


def forward(w, q, o):
    n = q.shape[0]
    focused_q = question_focus(w, q)
    fb = bag(w, focused_q)
    whole = bag(w, q)
    qb = w['knobs'][7] * whole + (1 - w['knobs'][7]) * fb
    ob = bag(w, o.reshape(-1, o.shape[-1])).reshape(n, 4, -1)
    novel = 1 - w['knobs'][0] * ob.mean(1, keepdim=True)
    ow = ob * w['idf'] * novel * (1 - w['knobs'][1] * qb.unsqueeze(1))
    qw = qb * w['idf']
    qk = affinity(qw @ w['key'].t(), w['keymass'])
    qv = affinity(qw @ w['value'].t(), w['valuemass'])
    clashes = ow[:, :, w['clash_src']] * (ob[:, :, w['clash_dst']] == 0)
    ov = ow @ w['value'].t() - w['knobs'][10] * (clashes @ w['value_clash'])
    ok = ow @ w['key'].t() - w['knobs'][10] * (clashes @ w['key_clash'])
    option_scale = torch.sqrt(ow.sum(-1, keepdim=True) + 3)
    ov = ov.clamp_min(0) / torch.sqrt(w['valuemass'] + 2) / option_scale
    ok = ok.clamp_min(0) / torch.sqrt(w['keymass'] + 2) / option_scale
    directional = qk.unsqueeze(1).square() * ov.pow(w['knobs'][6])
    reverse = w['knobs'][2] * qv.unsqueeze(1).square() * ok.pow(w['knobs'][6])
    memory = torch.maximum(directional, reverse) * w['strength']
    strongest = memory.topk(3, dim=-1).values
    score = strongest[:, :, 0] + w['knobs'][3] * strongest[:, :, 1] + w['knobs'][4] * strongest[:, :, 2]
    negative = request_negation(w, focused_q)
    score = score * (1 - 2 * negative.unsqueeze(1))
    score = score + ordered_properties(w, whole, ob) * comparison_polarity(w, focused_q).unsqueeze(1)
    score = score + arithmetic(w, q, o, fb)
    return score
