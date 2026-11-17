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


def add_literal_phrases(w, b, sequence, width):
    code = torch.zeros_like(sequence[:, :sequence.shape[1] - width + 1])
    valid = torch.ones_like(code, dtype=torch.bool)
    for offset in range(width):
        part = sequence[:, offset:offset + code.shape[1]]
        code = code * w['literal_base'] + part
        valid = valid * (part > 0) * (part < w['literal_base'])
    location = torch.searchsorted(w['literal_codes'], code.contiguous())
    location = location.clamp_max(w['literal_codes'].shape[0] - 1)
    valid = valid * (w['literal_codes'][location] == code)
    phrase = w['literal_ids'][location] * valid.to(torch.int64)
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
    literal_ids = w['literal_lex'][tokens]
    literal_ids = torch.where((literal_ids == 0) * (w['skip'][tokens] == 0), w['literal_base'], literal_ids)
    literal_sequence = literal_ids.gather(1, order)
    b = add_literal_phrases(w, b, literal_sequence, 2)
    b = add_literal_phrases(w, b, literal_sequence, 3)
    b = add_literal_phrases(w, b, literal_sequence, 4)
    b[:, 0] = 0
    b = b.clamp_max(1)
    b[:, w['override_dst']] = b[:, w['override_dst']] * (1 - b[:, w['override_src']])
    states = b[:, w['quantity_state']]
    ambiguous = (states.sum(-1, keepdim=True) > 1).expand_as(states)
    suppress = torch.zeros_like(b)
    indices = w['quantity_state'].reshape(1, -1).expand(tokens.shape[0], -1)
    suppress.scatter_add_(1, indices, ambiguous.reshape(tokens.shape[0], -1).to(torch.float32))
    return b * (suppress == 0)


def denied_concepts(w, tokens):
    # Interpret only short, single-negation clauses. Preserve barriers between
    # disjoint spans so masking cannot manufacture a multiword assertion.
    position = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    following_exception = w['option_neg_exception'][tokens].roll(-1, dims=-1)
    following_exception[:, -1] = 0
    cue = (w['option_negator'][tokens] > 0) * (following_exception == 0)
    boundary = w['option_neg_boundary'][tokens] > 0
    last_boundary = torch.where(boundary, position, -1).cummax(-1).values
    last_cue = torch.where(cue, position, -1).cummax(-1).values
    prefix = cue.to(torch.int64).cumsum(-1)
    boundary_count = prefix.gather(1, last_boundary.clamp_min(0))
    boundary_count = torch.where(last_boundary >= 0, boundary_count, 0)
    clause_count = prefix - boundary_count
    uncertain = (clause_count > 1).sum(-1) > 0
    content_count = (w['lex'][tokens] > 0).to(torch.int64).cumsum(-1)
    cue_content = content_count.gather(1, last_cue.clamp_min(0))
    close = (position - last_cue <= w['option_neg_limits'][0]) * (content_count - cue_content <= w['option_neg_limits'][1])
    inside = (clause_count == 1) * (last_cue > last_boundary) * close * (~cue)
    negative_tokens = torch.where(inside, tokens, torch.ones_like(tokens))
    negative_tokens = torch.where(cue, torch.zeros_like(tokens), negative_tokens)
    positive_tokens = torch.where((~inside) * (~cue), tokens, torch.ones_like(tokens))
    negative = bag(w, negative_tokens)
    positive = bag(w, positive_tokens)
    # A repeated concept affirmed elsewhere in the same option is ambiguous,
    # not evidence that every occurrence is denied.
    return negative * (1 - positive) * (~uncertain).unsqueeze(-1), uncertain


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
    direct = boundary_count == 0
    following_exception = w['option_neg_exception'][focused].roll(-1, dims=-1)
    following_exception[:, -1] = 0
    direct_not = w['not_word'][focused] * direct * (following_exception == 0)
    # Embedded 'incorrect/unlikely/least' describes a situation just as an
    # embedded 'not' does. Explicit EXCEPT may still end a conditional request.
    exclusion = w['global_exclusion'][focused]
    scoped = (w['neg'][focused] - exclusion) * direct
    count = (exclusion + scoped + direct_not).sum(-1)
    return count.remainder(2)


def comparison_polarity(w, focused):
    # 'Least' is already represented by the descending ordering cue. A direct
    # 'not' or 'except' reverses a comparison; 'least' alone must not do so twice.
    negative = request_negation(w, focused)
    direct = w['scope_boundary'][focused].cumsum(-1) == 0
    scoped_least = (w['least'][focused] * direct).sum(-1)
    negative = (negative - scoped_least).remainder(2)
    return 1 - 2 * negative


def numerical_choice_mask(w, o):
    # A point estimate must not silently stand for an inequality, an unknown
    # word/unit, or a quantity in an explicitly unsupported dimension.
    negated = w['option_negator'][o].sum(-1)
    bounded = w['numeric_bound_word'][o].sum(-1)
    unsupported = w['numeric_unsupported_unit'][o].sum(-1)
    unknown = (o == 1).sum(-1)
    return (negated + bounded + unsupported + unknown) == 0


def numerical_question_mask(w, q):
    # Do not turn an input inequality into an exact datum. Bound cues attach
    # through short grammatical spans to actual physical/cycle/atomic readings,
    # not to unrelated counts such as 'at least three students'.
    number = w['number'][q] >= 0
    position = torch.arange(q.shape[1], device=q.device).unsqueeze(0)
    unit = (physical_units(w, q).sum(-1) > 0) + (w['cycle_unit'][q] > 0) + (w['atom_words'][q][:, :, 3:].sum(-1) > 0)
    measured = torch.zeros_like(number)
    bridge = w['unit_bridge'][q] > 0
    for gap in range(1, 4):
        intervening = torch.ones_like(number)
        for inner in range(gap - 1):
            intervening = intervening * bridge.roll(-inner - 1, dims=-1)
        measured = measured + number * unit.roll(-gap, dims=-1) * intervening * (position < q.shape[1] - gap)
    marker = atom_markers(w, q).sum(-1) > 0
    atom_bridge = w['atom_bridge'][q] > 0
    for gap in range(1, 6):
        intervening = torch.ones_like(number)
        for inner in range(gap - 1):
            intervening = intervening * atom_bridge.roll(inner + 1, dims=-1)
        measured = measured + number * marker.roll(gap, dims=-1) * intervening * (position >= gap)
    following = q.roll(-1, dims=-1)
    following[:, -1] = 0
    up_to = (q == w['numeric_bound_up_to'][0]) * (following == w['numeric_bound_up_to'][1])
    cue = ((w['numeric_bound_word'][q] > 0) * (q != w['numeric_bound_up_to'][0])) + up_to
    postfix = w['numeric_bound_postfix'][q] > 0
    before_bridge = (w['numeric_bound_bridge'][q] > 0) + number
    after_bridge = (w['numeric_bound_tail'][q] > 0) + bridge + unit
    bounded = torch.zeros_like(number)
    for gap in range(1, 5):
        before = torch.ones_like(number)
        after = torch.ones_like(number)
        for inner in range(gap - 1):
            before = before * before_bridge.roll(inner + 1, dims=-1)
            after = after * after_bridge.roll(-inner - 1, dims=-1)
        bounded = bounded + cue.roll(gap, dims=-1) * before * (position >= gap)
        bounded = bounded + postfix.roll(-gap, dims=-1) * after * (position < q.shape[1] - gap)
    return (bounded * measured).sum(-1) == 0


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


def unit_readings(w, q):
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
    readings = units * preceding.clamp_min(0).unsqueeze(-1) * valid_unit
    ambiguous = ((numbers[:, :-1] >= 0) * (numbers[:, 1:] >= 0)).sum(-1) > 0
    return readings, valid_unit, ambiguous


def measurements(w, q):
    readings, valid, ambiguous = unit_readings(w, q)
    counts = valid.sum(1)
    values = readings.sum(1) / counts.clamp_min(1)
    values = torch.cat((values, torch.ones((q.shape[0], 1), device=q.device)), dim=-1)
    known = torch.cat(((counts == 1), torch.ones((q.shape[0], 1), device=q.device, dtype=torch.bool)), dim=-1)
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
    # An explicit quotient is not a product with the same unit names. Slash
    # punctuation is lost by TOKEN, but a retained 'per' must be respected.
    last = torch.where(units > 0, positions, -1).amax(-2)
    per = w['per_word'][o] > 0
    per_count = per.sum(-1)
    per_first = torch.where(per, positions.squeeze(-1), o.shape[-1]).amin(-1)
    before_per = last.unsqueeze(2).unsqueeze(3) < per_first.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    after_per = first.unsqueeze(2).unsqueeze(3) > per_first.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    correct_side = (((powers <= 0) + before_per) * ((powers >= 0) + after_per)).amin(-1)
    quotient = correct_side * (powers < 0).sum(-1).gt(0) * (per_count == 1).unsqueeze(-1).unsqueeze(-1)
    matches = matches * ((per_count == 0).unsqueeze(-1).unsqueeze(-1) + quotient)
    unitless = (present.sum(-1) == 0) * (per_count == 0)
    first_pattern = torch.arange(powers.shape[1], device=o.device) == 0
    matches = matches + unitless.unsqueeze(-1).unsqueeze(-1) * first_pattern
    conversion = scale.unsqueeze(2).unsqueeze(3).pow(powers).prod(-1)
    conversion = (conversion * matches).sum(-1) / matches.sum(-1).clamp_min(1)
    compatible = matches.sum(-1) > 0
    return conversion, compatible, unitless


def requested_unit_tokens(w, q, focused):
    # Only a contiguous unit expression after a focused 'in' is an output
    # request. 'In 3 seconds' remains an input datum; 'in a box' is not amperes.
    position = torch.arange(q.shape[1], device=q.device).unsqueeze(0)
    unsupported = (w['numeric_unsupported_unit'][q] > 0) + (w['cand_speed_units'][q] > 0)
    unit = ((physical_units(w, q).sum(-1) > 0) + unsupported) * (w['output_unit_excluded_symbol'][q] == 0)
    allowed = unit + (w['output_unit_bridge'][q] > 0)
    cue = (w['output_unit_cue'][q] > 0) * (focused > 0)
    running = cue
    found = torch.zeros_like(cue)
    selected = torch.zeros_like(cue)
    ambiguous = torch.zeros_like(cue)
    alternative = w['output_unit_alternative'][q] > 0
    for gap in range(1, 9):
        unknown = q.roll(-gap, dims=-1) == 1
        competing = alternative.roll(-gap, dims=-1) * allowed.roll(-gap - 1, dims=-1)
        ambiguous = ambiguous + running * (unknown + competing) * (position < q.shape[1] - gap)
        running = running * allowed.roll(-gap, dims=-1) * (position < q.shape[1] - gap)
        found = found + running * unit.roll(-gap, dims=-1)
        selected = selected + running.roll(gap, dims=-1)
    overflow = running * allowed.roll(-9, dims=-1) * (position < q.shape[1] - 9)
    count = found.sum(-1)
    tokens = q * selected.to(torch.int64)
    unsafe_units = w['numeric_unsupported_unit'][tokens].sum(-1) + (w['cand_speed_units'][tokens] > 0).sum(-1)
    repeated = (physical_units(w, tokens) > 0).sum(1).amax(-1) > 1
    safe = (count <= 1) * (overflow.sum(-1) == 0) * (unsafe_units == 0) * (ambiguous.sum(-1) == 0) * (~repeated)
    return tokens, count > 0, safe


def answer_measurements(w, q, focused, o):
    tokens, requested, safe = requested_unit_tokens(w, q, focused)
    default_conversion, default_compatible, default_bare = option_measurement(w, tokens.unsqueeze(1))
    conversion, compatible, bare = option_measurement(w, o)
    use_default = bare.unsqueeze(-1) * requested.reshape(-1, 1, 1)
    conversion = torch.where(use_default, default_conversion, conversion)
    compatible = compatible * ((~requested).reshape(-1, 1, 1) + default_compatible) * safe.reshape(-1, 1, 1)
    return conversion, compatible, bare, tokens, requested, safe


def volume_answer_units(w, o, bare, tokens, requested, safe):
    # Geometric/displaced volumes use mL internally, independently of the
    # density-equation templates. Respect a requested liter/cubic-length unit.
    output_units = physical_units(w, tokens)
    output_present = (output_units > 0).sum(1) > 0
    output_volume = (output_present.sum(-1) == 1) * output_present[:, 1] * (w['per_word'][tokens].sum(-1) == 0)
    output_scale = output_units[:, :, 1].amax(-1)
    default_scale = torch.where(requested, output_scale, torch.ones_like(output_scale))
    units = physical_units(w, o) * (w['skip'][o] == 0).unsqueeze(-1)
    present = units.sum(-2) > 0
    compatible = (((present.sum(-1) == 1) * present[:, :, 1]) + bare) * (w['per_word'][o].sum(-1) == 0)
    compatible = compatible * ((~requested) + output_volume).unsqueeze(-1) * safe.unsqueeze(-1)
    scale = torch.where(bare, default_scale.unsqueeze(-1), units[:, :, :, 1].amax(-1))
    return scale, compatible


def cycle_values(w, tokens):
    numbers = w['number'][tokens]
    previous = numbers.roll(1, dims=-1)
    previous[:, 0] = -1
    unit = w['cycle_unit'][tokens] > 0
    valid = unit * (previous >= 0)
    # A request for one cycle does not replace an observed multi-cycle count.
    nonunitary = valid * (previous != 1)
    valid = torch.where((nonunitary.sum(-1) > 0).unsqueeze(-1), nonunitary, valid)
    count = valid.sum(-1)
    cycles = (previous.clamp_min(0) * valid).sum(-1) / count.clamp_min(1)
    known = (count == 1) * (cycles > 0)
    each_before = w['each_word'][tokens].roll(1, dims=-1)
    each_before[:, 0] = 0
    period = (w['period_word'][tokens].sum(-1) > 0) + ((unit * (each_before > 0)).sum(-1) > 0)
    return cycles, known, period


def period_markers(w, tokens):
    each_before = w['each_word'][tokens].roll(1, dims=-1)
    each_before[:, 0] = 0
    return (w['period_word'][tokens] > 0) + ((w['cycle_unit'][tokens] > 0) * (each_before > 0))


def assigned_periods(w, q, time_readings):
    # Link a period/single-cycle cue to a time reading only through a short
    # grammatical span. An observation duration is not automatically a period.
    position = torch.arange(q.shape[1], device=q.device)
    bridge = (w['period_bridge'][q] > 0) + (w['number'][q] >= 0)
    previous_number = w['number'][q].roll(1, dims=-1)
    previous_number[:, 0] = -1
    multicycle = (w['cycle_unit'][q] > 0) * (previous_number >= 0) * (previous_number != 1)
    barrier = ((bridge == 0) + multicycle).to(torch.int64)
    inclusive = barrier.cumsum(-1)
    exclusive = inclusive - barrier
    earlier = torch.minimum(position.unsqueeze(0), position.unsqueeze(1))
    later = torch.maximum(position.unsqueeze(0), position.unsqueeze(1))
    intervening = exclusive[:, later] - inclusive[:, earlier]
    distance = (position.unsqueeze(0) - position.unsqueeze(1)).abs()
    linked = (intervening <= 0) * (distance <= 8) * time_readings.unsqueeze(1)
    return period_markers(w, q) * (linked.sum(-1) > 0)


def arithmetic_request(w, focused):
    ids = w['lex'][focused]
    single = w['single_feature'][ids]
    triggers = w['math_trigger'][:, ids] + w['math_trigger'][:, single]
    position = torch.arange(focused.shape[1], device=focused.device).reshape(1, -1, 1)
    first = torch.where(triggers.permute(1, 2, 0) > 0, position, focused.shape[1]).amin(1)
    previous_how = w['how_word'][focused].roll(1, dims=-1)
    previous_how[:, 0] = 0
    how_long = (previous_how * w['long_word'][focused]) > 0
    duration_first = torch.where(how_long.unsqueeze(-1) * w['duration_rule'] > 0, position, focused.shape[1]).amin(1)
    first = torch.minimum(first, duration_first)
    return (first == first.amin(-1, keepdim=True)) * (first < focused.shape[1])


def arithmetic(w, q, o, focused_bag):
    values, known, ambiguous = measurements(w, q)
    lhs = values[:, w['math_left']]
    rhs = values[:, w['math_right']]
    divisor = torch.where((rhs == 0) * (w['math_op'] < 0), torch.ones_like(rhs), rhs)
    prediction = lhs * divisor.pow(w['math_op'])
    active = known[:, w['math_left']] * known[:, w['math_right']]
    active = active * ((rhs > 0) + (w['math_op'] > 0) > 0)
    cycles, cycle_known, cycle_period = cycle_values(w, q)
    focused = question_focus(w, q)
    readings, valid_readings, reading_ambiguous = unit_readings(w, q)
    assigned = assigned_periods(w, q, valid_readings[:, :, 3])
    period_given = (assigned.sum(-1) > 0) * known[:, 3] * (values[:, 3] > 0)
    period_requested = (period_markers(w, focused) * (~assigned)).sum(-1) > 0
    count_time = cycle_known * known[:, 3] * (values[:, 3] > 0) * (~period_given)
    count_frequency = cycle_known * known[:, 10] * (values[:, 10] > 0)
    wave_known = known[:, 10] * known[:, 2] * (w['wavelength_word'][q].sum(-1) > 0)
    period_from_time = (count_time * period_requested).unsqueeze(-1) * w['cycle_period_rule']
    frequency_from_count = count_time.unsqueeze(-1) * w['cycle_frequency_rule']
    total_from_frequency = (count_frequency * (~period_requested)).unsqueeze(-1) * w['cycle_period_rule']
    direct_period = (period_given * period_requested).unsqueeze(-1) * w['cycle_period_rule']
    total_from_period = (period_given * cycle_known * (~period_requested)).unsqueeze(-1) * w['cycle_period_rule']
    wave_speed = wave_known.unsqueeze(-1) * w['wave_speed_rule']
    prediction = torch.where(total_from_frequency > 0, (cycles / values[:, 10].clamp_min(0.000001)).unsqueeze(-1), prediction)
    prediction = torch.where(period_from_time > 0, (values[:, 3] / cycles.clamp_min(0.000001)).unsqueeze(-1), prediction)
    prediction = torch.where(direct_period > 0, values[:, 3].unsqueeze(-1), prediction)
    prediction = torch.where(total_from_period > 0, (cycles * values[:, 3]).unsqueeze(-1), prediction)
    prediction = torch.where(frequency_from_count > 0, (cycles / values[:, 3].clamp_min(0.000001)).unsqueeze(-1), prediction)
    prediction = torch.where(wave_speed > 0, (values[:, 10] * values[:, 2]).unsqueeze(-1), prediction)
    active = (active + period_from_time + frequency_from_count + direct_period + total_from_period + wave_speed) > 0
    frequency_guard = (w['cycle_frequency_rule'] == 0) + (cycle_known + period_given).unsqueeze(-1)
    time_available = (period_requested * (known[:, 10] + count_time + period_given)) + ((~period_requested) * (count_frequency + period_given * cycle_known))
    period_guard = (w['cycle_period_rule'] == 0) + time_available.unsqueeze(-1)
    predicted_frequency = torch.where(period_given, 1 / values[:, 3].clamp_min(0.000001), cycles / values[:, 3].clamp_min(0.000001))
    frequency_agrees = (~(known[:, 10] * (period_given + count_time))) + ((predicted_frequency - values[:, 10]).abs() <= 0.0001 * values[:, 10].abs().clamp_min(1))
    wave_equation = w['cycle_period_rule'] + w['cycle_frequency_rule'] + w['wave_speed_rule']
    consistency = (wave_equation == 0) + frequency_agrees.unsqueeze(-1)
    work_guard = (w['force_work_rule'] == 0) + (w['work_forbid'][q].sum(-1) == 0).unsqueeze(-1)
    active = active * (frequency_guard > 0) * (period_guard > 0) * (consistency > 0) * (work_guard > 0)
    request = arithmetic_request(w, focused)
    active = active * request * (~ambiguous).unsqueeze(1)
    option_numbers = w['number'][o]
    numeric_count = (option_numbers >= 0).sum(-1)
    first = torch.where(option_numbers >= 0, torch.arange(o.shape[-1], device=o.device), o.shape[-1]).argmin(-1, keepdim=True)
    numeric_value = option_numbers.gather(-1, first).squeeze(-1).clamp_min(0)
    conversion, dimensional_match, bare, output_tokens, output_requested, output_safe = answer_measurements(w, q, focused, o)
    observed = numeric_value.unsqueeze(-1) * conversion
    error = (observed - prediction.unsqueeze(1)).abs() / (prediction.abs().unsqueeze(1) + 1)
    compatible = (numeric_count == 1).to(torch.float32)
    enough = (compatible.sum(1) >= 3).to(torch.float32)
    contribution = torch.exp(-w['knobs'][9] * error) * active.unsqueeze(1) * dimensional_match
    upper_prediction = torch.where(active, prediction, 0).amax(-1)
    lower_prediction = torch.where(active, prediction, torch.inf).amin(-1)
    predictions_agree = (upper_prediction - lower_prediction).abs() <= 0.0001 * upper_prediction.abs().clamp_min(1)
    # Equivalent formulas are constraints, not independent votes. Conflicting
    # predictions for the same requested quantity cause conservative abstention.
    return w['knobs'][8] * contribution.amax(-1) * compatible * enough.unsqueeze(1) * predictions_agree.unsqueeze(-1)


def displacement_arithmetic(w, q, o, focused_bag):
    readings, valid, ambiguous = unit_readings(w, q)
    volume_readings = readings[:, :, 1]
    volume_valid = valid[:, :, 1]
    upper = torch.where(volume_valid, volume_readings, 0).amax(-1)
    lower = torch.where(volume_valid, volume_readings, torch.inf).amin(-1)
    displacement = (upper - lower).clamp_min(0)
    direct = w['displacement_word'][q].sum(-1) > 0
    cylinder = (w['cylinder_word'][q].sum(-1) > 0) * (w['water_word'][q].sum(-1) > 0) * (w['object_word'][q].sum(-1) > 0)
    context = (direct + cylinder) * (w['displacement_forbid'][q].sum(-1) == 0)
    measured = (volume_valid.sum(-1) == 2) * (displacement > 0) * (~ambiguous) * context
    focused = question_focus(w, q)
    position = torch.arange(q.shape[1], device=q.device).unsqueeze(0)
    volume_first = torch.where(w['volume_request_word'][focused] > 0, position, q.shape[1]).amin(-1)
    density_first = torch.where(w['density_request_word'][focused] > 0, position, q.shape[1]).amin(-1)
    wants_density = (focused_bag @ w['density_query'] > 0) * (density_first < volume_first)
    wants_volume = (focused_bag @ w['volume_query'] > 0) * (volume_first < density_first)
    mass_known = valid[:, :, 0].sum(-1) == 1
    density = readings[:, :, 0].sum(-1) / displacement.clamp_min(0.000001)
    numbers = w['number'][o]
    numeric_count = (numbers >= 0).sum(-1)
    numeric = numbers.amax(-1).clamp_min(0)
    enough = (numeric_count == 1).sum(-1, keepdim=True) >= 3
    conversion, compatible, bare, output_tokens, output_requested, output_safe = answer_measurements(w, q, focused, o)
    volume_scale, volume_compatible = volume_answer_units(w, o, bare, output_tokens, output_requested, output_safe)
    observed_volume = numeric * volume_scale
    observed_density = numeric * conversion[:, :, 0]
    volume_error = (observed_volume - displacement.unsqueeze(-1)).abs() / (displacement.unsqueeze(-1) + 1)
    density_error = (observed_density - density.unsqueeze(-1)).abs() / (density.unsqueeze(-1) + 1)
    volume_score = torch.exp(-w['knobs'][9] * volume_error) * volume_compatible * wants_volume.unsqueeze(-1)
    density_score = torch.exp(-w['knobs'][9] * density_error) * compatible[:, :, 0] * (wants_density * mass_known).unsqueeze(-1)
    return w['knobs'][8] * (volume_score + density_score) * measured.unsqueeze(-1) * (numeric_count == 1) * enough


def geometric_arithmetic(w, q, o, focused_bag):
    readings, valid, ambiguous = unit_readings(w, q)
    lengths = readings[:, :, 2]
    length_valid = valid[:, :, 2]
    count = length_valid.sum(-1)
    rectangular = (w['geometry_rectangle'][q].sum(-1) > 0) * (count == 3)
    cube = (w['geometry_cube'][q].sum(-1) > 0) * (w['geometry_edge'][q].sum(-1) > 0) * (count == 1)
    product = torch.where(length_valid, lengths, torch.ones_like(lengths)).prod(-1)
    volume = torch.where(cube, product.pow(3), product) * 1000000
    active = (rectangular + cube) * (~ambiguous) * (w['geometry_forbid'][q].sum(-1) == 0) * (volume > 0)
    focused = question_focus(w, q)
    position = torch.arange(q.shape[1], device=q.device).unsqueeze(0)
    volume_first = torch.where(w['volume_request_word'][focused] > 0, position, q.shape[1]).amin(-1)
    density_first = torch.where(w['density_request_word'][focused] > 0, position, q.shape[1]).amin(-1)
    wants_density = (focused_bag @ w['density_query'] > 0) * (density_first < volume_first)
    wants_volume = (focused_bag @ w['volume_query'] > 0) * (volume_first < density_first)
    mass_known = valid[:, :, 0].sum(-1) == 1
    density = readings[:, :, 0].sum(-1) / volume.clamp_min(0.000001)
    numbers = w['number'][o]
    numeric_count = (numbers >= 0).sum(-1)
    numeric = numbers.amax(-1).clamp_min(0)
    enough = (numeric_count == 1).sum(-1, keepdim=True) >= 3
    conversion, compatible, bare, output_tokens, output_requested, output_safe = answer_measurements(w, q, focused, o)
    volume_scale, volume_compatible = volume_answer_units(w, o, bare, output_tokens, output_requested, output_safe)
    observed_volume = numeric * volume_scale
    observed_density = numeric * conversion[:, :, 0]
    volume_error = (observed_volume - volume.unsqueeze(-1)).abs() / (volume.unsqueeze(-1) + 1)
    density_error = (observed_density - density.unsqueeze(-1)).abs() / (density.unsqueeze(-1) + 1)
    volume_score = torch.exp(-w['knobs'][9] * volume_error) * volume_compatible * wants_volume.unsqueeze(-1)
    density_score = torch.exp(-w['knobs'][9] * density_error) * compatible[:, :, 0] * (wants_density * mass_known).unsqueeze(-1)
    return w['knobs'][8] * (volume_score + density_score) * active.unsqueeze(-1) * (numeric_count == 1) * enough


def scalar_count_choices(w, o, numbers, prediction, filler):
    count = (numbers >= 0).sum(-1)
    value = numbers.amax(-1).clamp_min(0)
    pure = ((numbers >= 0) + (filler[o] > 0) + (w['skip'][o] > 0)).amin(-1)
    compatible = (count == 1) * pure * numerical_choice_mask(w, o)
    enough = compatible.sum(-1, keepdim=True) == 4
    error = (value - prediction.unsqueeze(-1)).abs() / (prediction.abs().unsqueeze(-1) + 1)
    return w['count_numeric_gain'] * torch.exp(-w['count_numeric_error'] * error) * compatible * enough


def chromosome_arithmetic(w, q, o, focused, focused_bag):
    # Count chromosomes, not cells, DNA molecules, or chromosome pairs. The
    # explicit source datum must precede the selection request and be bound
    # through a short grammatical span to a chromosome marker.
    source_tokens = q * (focused == 0).to(torch.int64)
    source = bag(w, source_tokens)
    numbers = w['number'][q]
    source_numbers = w['number'][source_tokens]
    position = torch.arange(q.shape[1], device=q.device).unsqueeze(0)
    marker = w['count_chromosome_word'][q] > 0
    bridge = w['count_chromosome_bridge'][q] > 0
    attached = torch.zeros_like(marker)
    for gap in range(1, 5):
        before = torch.ones_like(marker)
        after = torch.ones_like(marker)
        for inner in range(gap - 1):
            before = before * bridge.roll(inner + 1, dims=-1)
            after = after * bridge.roll(-inner - 1, dims=-1)
        attached = attached + marker.roll(gap, dims=-1) * before * (position >= gap)
        attached = attached + marker.roll(-gap, dims=-1) * after * (position < q.shape[1] - gap)
    input_count = ((source_numbers >= 0) * attached).sum(-1)
    value = torch.where((source_numbers >= 0) * attached, source_numbers, 0).amax(-1)
    sd = source @ w['count_diploid'] > 0
    sh = source @ w['count_haploid'] > 0
    td = focused_bag @ w['count_diploid'] > 0
    th = focused_bag @ w['count_haploid'] > 0
    daughter = focused_bag @ w['count_daughter'] > 0
    mitosis = w['count_mitosis'][q].sum(-1) > 0
    meiosis = w['count_meiosis'][q].sum(-1) > 0
    focused_mitosis = w['count_mitosis'][focused].sum(-1) > 0
    focused_meiosis = w['count_meiosis'][focused].sum(-1) > 0
    reduce_count = sd * (~sh) * (th + daughter * meiosis) * (~td) * (~focused_mitosis)
    restore_count = sh * (~sd) * td * (~th) * (~focused_mitosis) * (~focused_meiosis)
    target_consistent = ((~td) + sd) * ((~th) + sh) * (~(td * th)) * (~(sd * sh))
    preserve_count = (source @ w['count_cell'] > 0) * daughter * mitosis * (~meiosis) * target_consistent
    modes = torch.stack((reduce_count, restore_count, preserve_count), dim=-1)
    prediction = value * (modes * w['count_factors']).sum(-1)
    request = focused_bag @ w['count_chromosome'] > 0
    forbidden = w['count_biology_forbidden'][q].sum(-1) + w['option_negator'][q].sum(-1) + w['numeric_bound_word'][q].sum(-1)
    active = request * (input_count == 1) * ((numbers >= 0).sum(-1) == 1) * (value > 0)
    active = active * (modes.sum(-1) == 1) * (prediction == prediction.round()) * (forbidden == 0) * (~(mitosis * meiosis))
    evidence = scalar_count_choices(w, o, w['number'][o], prediction, w['count_chromosome_filler'])
    return evidence * active.unsqueeze(-1)


def magnification_arithmetic(w, q, o, focused, whole, focused_bag):
    # A compound microscope's total magnification is eyepiece x objective.
    # Each input must attach unambiguously to its own explicitly named lens.
    numbers = w['count_magnification_number'][q]
    labels = w['count_magnification_label'][q]
    le, re, li, ri, ld, rd = nearby_entities(q, labels, w['count_magnification_bridge'], 5)
    nearest = torch.where(ld <= rd, le, re)
    unambiguous = (ld != rd) + ((le == re) * (le > 0))
    valid = (numbers > 0) * (nearest > 0) * unambiguous
    counts = torch.zeros((q.shape[0], 3), device=q.device)
    values = torch.zeros_like(counts)
    counts.scatter_add_(1, nearest, valid.to(torch.float32))
    values.scatter_add_(1, nearest, numbers.clamp_min(0) * valid)
    counts[:, 0] = 0
    values[:, 0] = 0
    prediction = values[:, 1] * values[:, 2]
    request = (focused_bag @ w['count_magnification'] > 0) * (whole @ w['count_microscope'] > 0)
    input_labels = ((labels == 1).sum(-1) == 1) * ((labels == 2).sum(-1) == 1)
    target_labels = w['count_magnification_label'][focused].sum(-1) == 0
    forbidden = w['count_magnification_forbidden'][q].sum(-1) + w['option_negator'][q].sum(-1) + w['numeric_bound_word'][q].sum(-1)
    active = request * input_labels * target_labels * (counts[:, 1:] == 1).amin(-1)
    readings, dimensioned, ambiguous = unit_readings(w, q)
    active = active * ((numbers >= 0).sum(-1) == 2) * (forbidden == 0) * (dimensioned.sum(dim=(1, 2)) == 0)
    evidence = scalar_count_choices(w, o, w['count_magnification_number'][o], prediction, w['count_magnification_filler'])
    return evidence * active.unsqueeze(-1)


def absolute_arithmetic(w, q, o, focused, whole, focused_bag):
    # Share question/choice applicability guards across every absolute solver.
    evidence = arithmetic(w, q, o, focused_bag)
    evidence = evidence + nuclear_arithmetic(w, q, o, focused, whole)
    evidence = evidence + displacement_arithmetic(w, q, o, focused_bag)
    evidence = evidence + geometric_arithmetic(w, q, o, focused_bag)
    evidence = evidence + kinetic_energy_arithmetic(w, q, o, focused, focused_bag)
    evidence = evidence + chromosome_arithmetic(w, q, o, focused, focused_bag)
    evidence = evidence + magnification_arithmetic(w, q, o, focused, whole, focused_bag)
    point_choices = numerical_choice_mask(w, o)
    # An unreadable or bounded alternative is not proved false merely because
    # three other options can be read as point quantities.
    return evidence * point_choices * point_choices.amin(-1, keepdim=True) * numerical_question_mask(w, q).unsqueeze(-1)


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
    # Unknown options receive neutral evidence, not an advantage over every
    # recognized option in a descending comparison. Centering also preserves
    # neutrality when an exclusion request reverses the ordering channel.
    center = (rank_values * known).sum(1, keepdim=True) / known.sum(1, keepdim=True).clamp_min(1)
    ranking = (rank_values - center) * known * (active * direction * enough).unsqueeze(1)
    return w['knobs'][5] * ranking.sum(-1)


def atom_markers(w, tokens):
    raw = w['atom_words'][tokens]
    previous = raw.roll(1, dims=1)
    previous[:, 0] = 0
    mass = raw[:, :, 2] * previous[:, :, 1] + raw[:, :, 1] * previous[:, :, 0]
    atomic = raw[:, :, 2] * previous[:, :, 0]
    return torch.cat((mass.unsqueeze(-1), atomic.unsqueeze(-1), raw[:, :, 3:]), dim=-1).clamp_max(1)


def nuclear_arithmetic(w, q, o, focused, whole):
    marker = atom_markers(w, q)
    numeric = w['number'][q]
    position = torch.arange(q.shape[1], device=q.device)
    distance = (position.unsqueeze(0) - position.unsqueeze(1)).abs().unsqueeze(0)
    barrier = (w['atom_bridge'][q] == 0).to(torch.int64)
    inclusive = barrier.cumsum(-1)
    exclusive = inclusive - barrier
    earlier = torch.minimum(position.unsqueeze(0), position.unsqueeze(1))
    later = torch.maximum(position.unsqueeze(0), position.unsqueeze(1))
    intervening = exclusive[:, later] - inclusive[:, earlier]
    eligible = (numeric >= 0).unsqueeze(1) * (intervening <= 0)
    distance = torch.where(eligible, distance, q.shape[1])
    nearest = distance.argmin(-1)
    closest = distance.amin(-1)
    unique = (distance == closest.unsqueeze(-1)).sum(-1) == 1
    measured = numeric.gather(1, nearest).clamp_min(0)
    valid = marker * ((closest <= 5) * unique).unsqueeze(-1)
    readings = (valid * measured.unsqueeze(-1)).sum(1) / valid.sum(1).clamp_min(1)
    upper = torch.where(valid > 0, measured.unsqueeze(-1), 0).amax(1)
    lower = torch.where(valid > 0, measured.unsqueeze(-1), torch.inf).amin(1)
    known = (valid.sum(1) > 0) * ((upper - lower).abs() < 0.01)
    conflicting = ((valid.sum(1) > 0) * ((upper - lower).abs() >= 0.01)).sum(-1) > 0
    neutral = (w['neutral_word'][q].sum(-1) > 0) * (w['ion_word'][q].sum(-1) == 0) * (w['electron_change'][q].sum(-1) == 0)
    proton = torch.where(known[:, 1], readings[:, 1], torch.where(known[:, 2], readings[:, 2], readings[:, 4]))
    proton_known = known[:, 1] + known[:, 2] + neutral * known[:, 4]
    derived_proton = torch.where(proton_known, proton, readings[:, 0] - readings[:, 3])
    derived_proton_known = proton_known + known[:, 0] * known[:, 3]
    neutron = torch.where(known[:, 3], readings[:, 3], readings[:, 0] - proton)
    neutron_known = known[:, 3] + known[:, 0] * proton_known
    mass = torch.where(known[:, 0], readings[:, 0], proton + readings[:, 3])
    mass_known = known[:, 0] + proton_known * known[:, 3]
    electron_known = derived_proton_known * neutral
    predictions = torch.stack((mass, derived_proton, neutron, derived_proton), dim=-1)
    available = torch.stack((mass_known, derived_proton_known, neutron_known, electron_known), dim=-1)
    focused_marker = atom_markers(w, focused)
    first = torch.where(focused_marker > 0, position.reshape(1, -1, 1), q.shape[1]).amin(1)
    target = w['atom_targets'][first.argmin(-1)]
    prediction = predictions.gather(1, target.unsqueeze(-1))
    active = available.gather(1, target.unsqueeze(-1))
    active = active * (first.amin(-1, keepdim=True) < q.shape[1]) * (whole @ w['atom_context'] > 0).unsqueeze(-1)
    ambiguous_numbers = ((numeric[:, :-1] >= 0) * (numeric[:, 1:] >= 0)).sum(-1, keepdim=True) > 0
    atomic_agrees = (~(known[:, 1] * known[:, 2])) + ((readings[:, 1] - readings[:, 2]).abs() < 0.01)
    mass_large = (~(known[:, 0] * proton_known)) + (readings[:, 0] >= proton)
    neutron_fits = (~(known[:, 0] * known[:, 3])) + (readings[:, 0] >= readings[:, 3])
    totals_agree = (~(known[:, 0] * known[:, 3] * proton_known)) + ((readings[:, 0] - proton - readings[:, 3]).abs() < 0.01)
    electron_agrees = (~(neutral * known[:, 4] * derived_proton_known)) + ((readings[:, 4] - derived_proton).abs() < 0.01)
    consistent = atomic_agrees * mass_large * neutron_fits * totals_agree * electron_agrees
    active = active * (prediction >= 0) * (~ambiguous_numbers) * consistent.unsqueeze(-1) * (~conflicting).unsqueeze(-1)
    active = active * (w['electron_change'][q].sum(-1, keepdim=True) == 0)
    option_numbers = w['number'][o]
    count = (option_numbers >= 0).sum(-1)
    value = option_numbers.amax(-1).clamp_min(0)
    physical = physical_units(w, o) * (w['skip'][o] == 0).unsqueeze(-1)
    unitless = physical.sum(dim=(-1, -2)) == 0
    compatible = (count == 1) * unitless
    compatible = compatible * (compatible.sum(-1, keepdim=True) >= 3)
    error = (value - prediction).abs() / (prediction.abs() + 1)
    return w['knobs'][8] * torch.exp(-w['knobs'][9] * error) * active * compatible


def logical_evidence(w, whole, ob, focused_bag, denied, uncertain):
    question_match = (whole @ w['logic_groups'].t() > 0).to(torch.float32)
    question_match[:, 0] = 1
    focus_match = (focused_bag @ w['logic_groups'].t() > 0).to(torch.float32)
    focus_match[:, 0] = 1
    needed = question_match[:, w['logic_required']].amin(-1)
    forbidden = question_match[:, w['logic_forbidden']].amax(-1)
    focused_need = focus_match[:, w['logic_focused_required']].amin(-1)
    focused_avoid = focus_match[:, w['logic_focused_forbidden']].amax(-1)
    active = needed * (1 - forbidden) * focused_need * (1 - focused_avoid)
    option_match = ((ob * (1 - denied)) @ w['logic_groups'].t() > 0).to(torch.float32)
    denied_match = (denied @ w['logic_groups'].t() > 0).to(torch.float32)
    option_match[:, :, 0] = 1
    denied_match[:, :, 0] = 1
    asserted_positive = option_match[:, :, w['logic_positive']].amin(-1) * w['logic_has_positive']
    asserted_negative = option_match[:, :, w['logic_negative']].amin(-1) * w['logic_has_negative']
    denied_positive = denied_match[:, :, w['logic_positive']].amin(-1) * w['logic_has_positive']
    denied_negative = denied_match[:, :, w['logic_negative']].amin(-1) * w['logic_has_negative']
    support = torch.maximum(asserted_positive, denied_negative)
    contradiction = torch.maximum(asserted_negative, denied_positive)
    confidence = active.unsqueeze(1) * w['logic_confidence'] * (~uncertain).unsqueeze(-1)
    # Max pooling prevents overlapping declarations from counting as independent
    # evidence. Explicit incompatible assertions provide negative evidence.
    positive = (support * confidence).amax(-1)
    negative = (contradiction * confidence).amax(-1)
    return positive - negative


def assertion_evidence(w, whole, ob, focused, denied, uncertain):
    question_match = (whole @ w['claim_groups'].t() > 0).to(torch.float32)
    option_match = (ob @ w['claim_groups'].t() > 0).to(torch.float32)
    question_match[:, 0] = 1
    option_match[:, :, 0] = 1
    q_subject = question_match[:, w['claim_subject']].amin(-1)
    o_subject = option_match[:, :, w['claim_subject']].amin(-1)
    # An explicit option subject is more specific than an inherited question
    # subject. Requirements cannot be assembled from fragments of two subjects.
    subject = torch.maximum(w['role_gains'][0] * q_subject.unsqueeze(1), o_subject)
    union = torch.maximum(question_match.unsqueeze(1), option_match)
    context = union[:, :, w['claim_context']].amin(-1)
    # Required groups use true padding; forbidden groups must explicitly mask
    # that same padding so an empty exclusion list never blocks a schema.
    forbidden = (union[:, :, w['claim_forbidden_context']] * (w['claim_forbidden_context'] != 0)).amax(-1)
    context = context * (1 - forbidden)
    affirmed_match = ((ob * (1 - denied)) @ w['claim_groups'].t() > 0).to(torch.float32)
    denied_match = (denied @ w['claim_groups'].t() > 0).to(torch.float32)
    affirmed_match[:, :, 0] = 1
    denied_match[:, :, 0] = 1
    asserted_positive = affirmed_match[:, :, w['claim_positive']].amin(-1) * w['claim_has_positive']
    asserted_negative = affirmed_match[:, :, w['claim_negative']].amin(-1) * w['claim_has_negative']
    denied_positive = denied_match[:, :, w['claim_positive']].amin(-1) * w['claim_has_positive']
    denied_negative = denied_match[:, :, w['claim_negative']].amin(-1) * w['claim_has_negative']
    positive = torch.maximum(asserted_positive, denied_negative)
    negative = torch.maximum(asserted_negative, denied_positive)
    confidence = subject * context * w['claim_strength'] * (~uncertain).unsqueeze(-1)
    support = (positive * confidence).amax(-1)
    contradiction = (negative * confidence).amax(-1)
    boundary_count = w['scope_boundary'][focused].cumsum(-1)
    absence = ((w['absence_word'][focused] > 0) * (boundary_count == 0)).sum(-1) > 0
    return (support - contradiction) * (1 - 2 * absence.to(torch.float32).unsqueeze(-1))


def experimental_roles(w, q, ob, focused_bag):
    # Parse two explicitly causal constructions: 'effect of X on Y' and
    # 'how X affects Y'. No relation is guessed from mere co-occurrence.
    position = torch.arange(q.shape[1], device=q.device)
    delta = position.unsqueeze(0) - position.unsqueeze(1)
    next_of = w['literal_of'][q].roll(-1, dims=-1)
    next_of[:, -1] = 0
    effect = (w['effect_word'][q] > 0) * (next_of > 0)
    on_after = (w['literal_on'][q] > 0).unsqueeze(1) * (delta > 1) * (delta <= 24)
    on_index = torch.where(on_after, position.reshape(1, 1, -1), q.shape[1]).amin(-1)
    effect_valid = effect * (on_index < q.shape[1])
    how_before = (w['literal_how'][q] > 0).unsqueeze(1) * (delta < -1) * (delta >= -24)
    how_index = torch.where(how_before, position.reshape(1, 1, -1), -1).amax(-1)
    affect_valid = (w['affect_verb'][q] > 0) * (how_index >= 0)
    valid = effect_valid + affect_valid
    selected = (valid * (position + 1)).argmax(-1, keepdim=True)
    starts = torch.where(effect_valid, position.unsqueeze(0) + 1, how_index)
    ends = torch.where(effect_valid, on_index, position.unsqueeze(0))
    start = starts.gather(1, selected)
    end = ends.gather(1, selected)
    boundary = (w['role_boundary'][q] > 0) * (position.unsqueeze(0) > end)
    stop = torch.where(boundary, position.unsqueeze(0), q.shape[1]).amin(-1, keepdim=True)
    x_mask = (position.unsqueeze(0) > start) * (position.unsqueeze(0) < end)
    y_mask = (position.unsqueeze(0) > end) * (position.unsqueeze(0) < stop) * (position.unsqueeze(0) <= end + 14)
    # A second, lower-confidence schema handles 'changes X and measures Y'.
    # Only active verbs in the experimental description, before the request,
    # qualify. Passive clauses and multiple candidate actions cause abstention.
    focused = question_focus(w, q)
    request_start = torch.where(focused > 0, position.unsqueeze(0), q.shape[1]).amin(-1, keepdim=True)
    passive = w['passive_auxiliary'][q].roll(1, dims=-1)
    passive[:, 0] = 0
    prior = (position.unsqueeze(0) < request_start) * (passive == 0)
    manipulated = (w['manipulate_verb'][q] > 0) * prior
    measured = (w['measure_verb'][q] > 0) * prior
    action_valid = (manipulated.sum(-1) == 1) * (measured.sum(-1) == 1)
    x_action = (manipulated * (position + 1)).argmax(-1, keepdim=True)
    y_action = (measured * (position + 1)).argmax(-1, keepdim=True)
    actions = manipulated + measured + (w['action_boundary'][q] > 0)
    x_stop = torch.where(actions * (position.unsqueeze(0) > x_action), position.unsqueeze(0), q.shape[1]).amin(-1, keepdim=True)
    y_stop = torch.where(actions * (position.unsqueeze(0) > y_action), position.unsqueeze(0), q.shape[1]).amin(-1, keepdim=True)
    x_action_mask = (position.unsqueeze(0) > x_action) * (position.unsqueeze(0) < x_stop) * (position.unsqueeze(0) <= x_action + 14)
    y_action_mask = (position.unsqueeze(0) > y_action) * (position.unsqueeze(0) < y_stop) * (position.unsqueeze(0) <= y_action + 14)
    explicit = valid.sum(-1) == 1
    fallback = (valid.sum(-1) == 0) * action_valid
    x_mask = torch.where(explicit.unsqueeze(-1), x_mask, x_action_mask)
    y_mask = torch.where(explicit.unsqueeze(-1), y_mask, y_action_mask)
    x_bag = bag(w, q * x_mask.to(torch.int64))
    y_bag = bag(w, q * y_mask.to(torch.int64))
    x_unique = x_bag * (1 - y_bag) * w['idf']
    y_unique = y_bag * (1 - x_bag) * w['idf']
    option_mass = (ob * w['idf']).sum(-1).clamp_min(1)
    x_overlap = (ob * x_unique.unsqueeze(1)).sum(-1)
    y_overlap = (ob * y_unique.unsqueeze(1)).sum(-1)
    x_similarity = x_overlap / torch.sqrt(option_mass * x_unique.sum(-1, keepdim=True).clamp_min(1))
    y_similarity = y_overlap / torch.sqrt(option_mass * y_unique.sum(-1, keepdim=True).clamp_min(1))
    independent = x_similarity - w['role_gains'][1] * y_similarity
    dependent = y_similarity - w['role_gains'][1] * x_similarity
    request = (focused_bag @ w['role_query'].t() > 0).to(torch.float32)
    enough = (x_overlap.amax(-1) > 0) * (y_overlap.amax(-1) > 0)
    active = (explicit + fallback) * enough * (request.sum(-1) == 1)
    gain = torch.where(explicit, w['role_gains'][2], w['role_gains'][3])
    score = independent * request[:, :1] + dependent * request[:, 1:]
    return gain.unsqueeze(-1) * score * active.unsqueeze(-1)


def relation_cues(w, tokens):
    sequence = w['slot_cue_lex'][tokens]
    found = torch.zeros((tokens.shape[0], w['slot_cue_groups'].shape[0]), device=tokens.device)
    for width in range(1, 4):
        code = torch.zeros_like(sequence[:, :sequence.shape[1] - width + 1])
        valid = torch.ones_like(code, dtype=torch.bool)
        for offset in range(width):
            part = sequence[:, offset:offset + code.shape[1]]
            code = code * w['slot_cue_base'] + part
            valid = valid * (part > 0)
        location = torch.searchsorted(w['slot_cue_codes'], code.contiguous())
        location = location.clamp_max(w['slot_cue_codes'].shape[0] - 1)
        valid = valid * (w['slot_cue_codes'][location] == code)
        cue_id = w['slot_cue_ids'][location] * valid.to(torch.int64)
        found.scatter_add_(1, cue_id, torch.ones_like(cue_id, dtype=torch.float32))
    found[:, 0] = 0
    groups = (found @ w['slot_cue_groups'] > 0).to(torch.float32)
    groups[:, 0] = 1
    return groups[:, w['slot_cue_required']].amin(-1)


def slot_evidence(w, o, focused, whole, ob, denied, uncertain):
    # Relation cues are independent of the entities: producing oxygen is not
    # the same slot as consuming oxygen. Inversion exchanges only the entities.
    qm = (whole @ w['slot_groups'].t() > 0).to(torch.float32)
    om = ((ob * (1 - denied)) @ w['slot_groups'].t() > 0).to(torch.float32)
    dm = (denied @ w['slot_groups'].t() > 0).to(torch.float32)
    qm[:, 0] = 1
    om[:, :, 0] = 1
    dm[:, :, 0] = 1
    qs = qm[:, w['slot_subject']].amin(-1)
    qv = qm[:, w['slot_value']].amin(-1)
    os = om[:, :, w['slot_subject']].amin(-1)
    ov = om[:, :, w['slot_value']].amin(-1)
    ds = dm[:, :, w['slot_subject']].amin(-1)
    dv = dm[:, :, w['slot_value']].amin(-1)
    cue = relation_cues(w, focused)[:, w['slot_relation']]
    option_cue = relation_cues(w, o.reshape(-1, o.shape[-1])).reshape(o.shape[0], 4, -1)
    option_cue = option_cue[:, :, w['slot_relation']]
    direct_cue = torch.maximum(cue.unsqueeze(1), 0.65 * option_cue)
    inverse_cue = torch.maximum(cue, w['slot_implicit'][w['slot_relation']])
    # Do not answer by copying a subject/value already supplied by the stem.
    # This also abstains on comparisons explicitly naming both sides of a fact.
    direct = qs.unsqueeze(1) * (1 - qv).unsqueeze(1) * direct_cue
    inverse = qv.unsqueeze(1) * (1 - qs).unsqueeze(1) * inverse_cue.unsqueeze(1)
    explicit = 0.6 * os * option_cue
    support = torch.maximum(torch.maximum(direct * ov, inverse * os), explicit * ov) * w['slot_strength']
    contradiction = torch.maximum(torch.maximum(direct * dv, inverse * ds), explicit * dv) * w['slot_strength']
    return (support.amax(-1) - contradiction.amax(-1)) * (~uncertain)


def nearby_entities(tokens, entities, bridges, limit):
    # All edges are attached through short grammatical spans. Clause words and
    # intervening content prevent one sentence from borrowing another's object.
    position = torch.arange(tokens.shape[1], device=tokens.device)
    delta = position.unsqueeze(1) - position.unsqueeze(0)
    barrier = (bridges[tokens] == 0).to(torch.int64)
    inclusive = barrier.cumsum(-1)
    exclusive = inclusive - barrier
    earlier = torch.minimum(position.unsqueeze(0), position.unsqueeze(1))
    later = torch.maximum(position.unsqueeze(0), position.unsqueeze(1))
    clear = exclusive[:, later] - inclusive[:, earlier] <= 0
    eligible = (entities > 0).unsqueeze(1) * clear * (delta.abs() <= limit)
    left_distance = torch.where(eligible * (delta > 0), delta.abs(), tokens.shape[1])
    right_distance = torch.where(eligible * (delta < 0), delta.abs(), tokens.shape[1])
    li = left_distance.argmin(-1)
    ri = right_distance.argmin(-1)
    ld = left_distance.amin(-1)
    rd = right_distance.amin(-1)
    le = entities.gather(1, li) * (ld < tokens.shape[1])
    re = entities.gather(1, ri) * (rd < tokens.shape[1])
    return le, re, li, ri, ld, rd


def food_web_evidence(w, q, o, focused, uncertain):
    # Construct a graph from the current question, not from a memorized food web.
    n = q.shape[0]
    size = w['eco_size'].shape[0]
    position = torch.arange(q.shape[1], device=q.device).unsqueeze(0)
    entity = w['eco_entity'][q]
    le, re, li, ri, ld, rd = nearby_entities(q, entity, w['eco_bridge'], w['eco_constants'][0])
    request_start = torch.where(w['eco_request'][q] > 0, position, -1).amax(-1, keepdim=True)
    described = position < request_start
    by_prefix = w['eco_passive'][q].cumsum(-1)
    passive = by_prefix.gather(1, ri) - by_prefix > 0
    next_on = w['eco_on'][q].roll(-1, dims=-1)
    next_on[:, -1] = 0
    predicate = (w['eco_predicate'][q] > 0) + ((w['eco_feeds'][q] > 0) * (next_on > 0))
    eater = torch.where(passive, re, le)
    food = torch.where(passive, le, re)
    valid = predicate * described * (eater > 0) * (food > 0) * (eater != food)
    edges = torch.zeros((n, size * size), device=q.device)
    edges.scatter_add_(1, food * size + eater, valid.to(torch.float32))
    edges = edges.reshape(n, size, size).clamp_max(1)
    edges[:, 0, :] = 0
    edges[:, :, 0] = 0
    influence = edges - edges.transpose(1, 2)
    # A condition before the final request, or after its 'if/when', can supply
    # one explicit population perturbation. Multiple perturbed species abstain.
    conditional = w['eco_conditional'][q].cumsum(-1)
    at_request = conditional.gather(1, request_start.clamp_min(0))
    event_zone = described + (conditional > at_request)
    el, er, eli, eri, eld, erd = nearby_entities(q, entity, w['eco_event_bridge'], w['eco_constants'][3])
    nearest = torch.where(eld <= erd, el, er)
    unambiguous = (eld != erd) + ((el == er) * (el > 0))
    up = w['eco_up'][q] + w['eco_event_up'][q]
    down = w['eco_down'][q] + w['eco_event_down'][q]
    change = (up - down).clamp(-1, 1) * event_zone * unambiguous * (nearest > 0)
    positive_events = torch.zeros((n, size), device=q.device)
    negative_events = torch.zeros_like(positive_events)
    positive_events.scatter_add_(1, nearest, (change > 0).to(torch.float32))
    negative_events.scatter_add_(1, nearest, (change < 0).to(torch.float32))
    positive_events[:, 0] = 0
    negative_events[:, 0] = 0
    perturbation = (positive_events - negative_events).clamp(-1, 1)
    # Count sources before signed cancellation, and reject conflicting events.
    one_source = ((positive_events + negative_events) > 0).sum(-1) == 1
    conflicting = ((positive_events > 0) * (negative_events > 0)).sum(-1) > 0
    unattached_event = (event_zone * ((up + down) > 0) * ((nearest == 0) + (~unambiguous))).sum(-1) > 0
    one_source = one_source * (~conflicting) * (~unattached_event)
    first = torch.bmm(perturbation.unsqueeze(1), influence).squeeze(1)
    second = torch.bmm(first.unsqueeze(1), influence).squeeze(1)
    # A two-link inference cannot override direct evidence or feed back to the
    # perturbed species. Opposing pathways cancel instead of receiving votes.
    second = second * (first == 0) * (perturbation == 0)
    effect = (first + w['eco_constants'][2] * second).clamp(-1, 1)
    effect = effect * (perturbation == 0)
    option_entity = w['eco_entity'][o]
    option_bag = torch.zeros((n, 4, size), device=q.device)
    option_bag.scatter_add_(2, option_entity, torch.ones_like(option_entity, dtype=torch.float32))
    option_bag[:, :, 0] = 0
    option_bag = option_bag.clamp_max(1)
    option_count = option_bag.sum(-1)
    option_effect = (option_bag * effect.unsqueeze(1)).sum(-1)
    o_up = (w['eco_up'][o] + w['eco_event_up'][o]).sum(-1) > 0
    o_down = (w['eco_down'][o] + w['eco_event_down'][o]).sum(-1) > 0
    o_sign = o_up.to(torch.float32) - o_down.to(torch.float32)
    request_zone = (focused > 0) * (~event_zone)
    q_up = (up * request_zone).sum(-1, keepdim=True) > 0
    q_down = (down * request_zone).sum(-1, keepdim=True) > 0
    q_sign = q_up.to(torch.float32) - q_down.to(torch.float32)
    direction = torch.where(o_sign != 0, o_sign, q_sign)
    named_choice = option_effect * direction * (option_count == 1)
    target = entity * request_zone
    target_bag = torch.zeros((n, size), device=q.device)
    target_bag.scatter_add_(1, target, torch.ones_like(target, dtype=torch.float32))
    target_bag[:, 0] = 0
    target_bag = target_bag.clamp_max(1)
    target_effect = (target_bag * effect).sum(-1, keepdim=True)
    target_known = target_bag.sum(-1, keepdim=True) == 1
    directional_choice = target_effect * o_sign * target_known * (option_count == 0)
    context = w['eco_population'][q].sum(-1) > 0
    supported = w['eco_unsupported'][q].sum(-1) == 0
    # An entity mention is not enough to support an arbitrary option assertion.
    option_words = (option_entity > 0) + (w['eco_option_filler'][o] > 0) + (w['eco_up'][o] > 0) + (w['eco_down'][o] > 0) + (w['eco_event_up'][o] > 0) + (w['eco_event_down'][o] > 0) + (o == 0)
    option_grammar = option_words.amin(-1)
    option_safe = (w['option_negator'][o].sum(-1) == 0) * (~uncertain) * (~(o_up * o_down)) * option_grammar
    active = one_source * context * supported * (valid.sum(-1) > 0)
    return w['eco_constants'][1] * (named_choice + directional_choice) * active.unsqueeze(-1) * option_safe


def ratio_readings(w, tokens):
    numbers = w['number'][tokens]
    position = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    next_times = w['ratio_times'][tokens].roll(-1, dims=-1)
    next_times[:, -1] = 0
    factor_before = torch.zeros_like(numbers)
    down_before = torch.zeros_like(numbers)
    up_before = torch.zeros_like(numbers)
    bridge = w['ratio_bridge'][tokens] > 0
    for gap in range(1, 6):
        intervening = torch.ones_like(bridge)
        for inner in range(gap - 1):
            intervening = intervening * bridge.roll(inner + 1, dims=-1)
        valid = (position >= gap) * intervening
        factor_before = factor_before + w['ratio_factor_word'][tokens].roll(gap, dims=-1) * valid
        down_before = down_before + w['ratio_down'][tokens].roll(gap, dims=-1) * valid
        up_before = up_before + w['ratio_up'][tokens].roll(gap, dims=-1) * valid
    numeric = (numbers > 0) * ((next_times + factor_before) > 0)
    value = torch.where(numeric, numbers.clamp_min(0), w['ratio_factor'][tokens])
    reciprocal = numeric * (factor_before > 0) * (down_before > 0) * (value > 1)
    value = torch.where(reciprocal, 1 / value.clamp_min(0.000001), value)
    # 'Three times less' and 'twice more' are not unambiguous point ratios.
    # A decrease by a factor is supported; a bare decrease by N times is not.
    comparative = w['ratio_comparative'][tokens] > 0
    near_comparative = torch.zeros_like(comparative)
    for gap in range(1, 3):
        before = comparative.roll(gap, dims=-1) * (position >= gap)
        after = comparative.roll(-gap, dims=-1) * (position + gap < tokens.shape[1])
        near_comparative = near_comparative + before + after
    ambiguous = (value > 0) * near_comparative
    ambiguous = ambiguous + (down_before > 0) * (value > 1) * (~reciprocal)
    ambiguous = ambiguous + (up_before > 0) * (value > 0) * (value < 1)
    ambiguous = ambiguous + ((up_before + down_before) > 0) * (value == 1)
    ambiguous = ambiguous + (factor_before > 0) * (down_before > 0) * (value > 0) * (value < 1) * (~numeric)
    return value, ambiguous


def proportional_evidence(w, q, o, focused, whole):
    # Ratios are attached to explicitly named quantities, then composed through
    # authored power laws. No numerical values or laws are estimated from data.
    n = q.shape[0]
    size = w['ratio_powers'].shape[1]
    position = torch.arange(q.shape[1], device=q.device).unsqueeze(0)
    request_start = torch.where(focused > 0, position, q.shape[1]).amin(-1, keepdim=True)
    condition = w['eco_conditional'][q].cumsum(-1)
    condition_at_request = condition.gather(1, request_start.clamp_max(q.shape[1] - 1))
    source_zone = (position < request_start) + (condition > condition_at_request)
    target_tokens = focused * (~source_zone).to(torch.int64)
    target = bag(w, target_tokens)
    requested = (target @ w['ratio_target_bag'].t() > 0).to(torch.float32)
    contextual = (whole @ w['ratio_context'].t() > 0).to(torch.float32)
    contextual = torch.maximum(contextual, 1 - w['ratio_context_needed'])
    quantity = w['ratio_quantity'][q]
    # In 'distance between [two] masses', masses is the preposition's object,
    # not the quantity being multiplied. Preserve that small explicit frame.
    distance_marker = quantity == w['ratio_distance_id']
    between_marker = w['ratio_between'][q] > 0
    embedded_mass = torch.zeros_like(distance_marker)
    for gap in range(2, 4):
        embedded_mass = embedded_mass + distance_marker.roll(gap, dims=-1) * between_marker.roll(gap - 1, dims=-1) * (position >= gap)
    embedded_mass = embedded_mass * (quantity == w['ratio_mass_id'])
    quantity = quantity * (~embedded_mass).to(torch.int64)
    linked_tokens = torch.where(embedded_mass, w['ratio_bridge_token'], q)
    le, re, li, ri, ld, rd = nearby_entities(linked_tokens, quantity, w['ratio_bridge'], w['ratio_constants'][0])
    nearest = torch.where(ld <= rd, le, re)
    unambiguous = (ld != rd) + ((le == re) * (le > 0))
    ratios, ambiguous_ratio = ratio_readings(w, q)
    valid = (ratios > 0) * source_zone * unambiguous * (nearest > 0) * (~ambiguous_ratio)
    observations = torch.zeros((n, size), device=q.device)
    observations.scatter_add_(1, nearest, valid.to(torch.float32))
    logs = torch.zeros_like(observations)
    logs.scatter_add_(1, nearest, torch.log(ratios.clamp_min(0.000001)) * valid)
    observations[:, 0] = 0
    logs[:, 0] = 0
    changed = logs.abs() > 0.00001
    powers = w['ratio_powers']
    log_prediction = logs @ powers.t()
    prediction = torch.exp(log_prediction)
    used = powers != 0
    covered = (changed.to(torch.float32) @ used.to(torch.float32).t()) == changed.sum(-1, keepdim=True)
    active = requested * contextual * covered * (changed.sum(-1, keepdim=True) > 0)
    aligned = w['work_forbid'][q].sum(-1, keepdim=True) == 0
    active = active * ((~w['ratio_alignment']) + aligned)
    # A mentioned but unquantified input change is not silently held constant.
    generic_change = (w['ratio_change'][q] > 0) * source_zone
    explained_change = torch.zeros_like(generic_change)
    ratio_bridge = w['ratio_bridge'][q] > 0
    for gap in range(1, 6):
        intervening = torch.ones_like(ratio_bridge)
        for inner in range(gap - 1):
            intervening = intervening * ratio_bridge.roll(-(inner + 1), dims=-1)
        following = valid.roll(-gap, dims=-1)
        same_quantity = (nearest == nearest.roll(-gap, dims=-1)) * (nearest > 0)
        explained_change = explained_change + following * same_quantity * intervening * (position + gap < q.shape[1])
    # Every generic change needs its own local, following multiplier. A factor
    # earlier in the sentence cannot explain a later unquantified change.
    unexplained_change = (generic_change * (~explained_change)).sum(-1) > 0
    unattached = ((ratios > 0) * (ratios != 1) * source_zone * (~valid)).sum(-1) > 0
    repeated = (observations > 1).sum(-1) > 0
    numbers = w['number'][q]
    adjacent_numbers = ((numbers[:, :-1] >= 0) * (numbers[:, 1:] >= 0)).sum(-1) > 0
    forbidden = (w['ratio_forbidden'][q].sum(-1) > 0) + ((ambiguous_ratio * source_zone).sum(-1) > 0)
    # A ratio is not an absolute answer. Existing measurements can make bare
    # numeric choices ambiguous, so let the dimensional helpers handle them.
    measured_units = physical_units(w, q) * (w['skip'][q] == 0).unsqueeze(-1)
    absolute_context = measured_units.sum(dim=(-1, -2)) > 0
    safe = (~unexplained_change) * (~unattached) * (~repeated) * (~adjacent_numbers) * (~forbidden) * (~absolute_context)
    active = active * safe.unsqueeze(-1)
    upper = torch.where(active > 0, prediction, 0).amax(-1, keepdim=True)
    lower = torch.where(active > 0, prediction, torch.inf).amin(-1, keepdim=True)
    agreement = (upper - lower).abs() <= 0.00001 * upper.abs().clamp_min(1)
    option_tokens = o.reshape(-1, o.shape[-1])
    option_ratios, ambiguous_option = ratio_readings(w, option_tokens)
    option_ratios = option_ratios.reshape(n, 4, -1)
    ambiguous_option = ambiguous_option.reshape(n, 4, -1).sum(-1) > 0
    ratio_count = (option_ratios > 0).sum(-1)
    option_value = option_ratios.amax(-1)
    option_numbers = w['number'][o]
    numeric_count = (option_numbers >= 0).sum(-1)
    bare_allowed = ((option_numbers >= 0) + (w['ratio_option_filler'][o] > 0) + (o == 0)).amin(-1)
    bare = (ratio_count == 0) * (numeric_count == 1) * bare_allowed
    option_value = torch.where(bare, option_numbers.amax(-1), option_value)
    bound_numbers = ((option_numbers >= 0) * (option_ratios <= 0)).sum(-1) == 0
    count_valid = ((ratio_count == 1) * bound_numbers) + bare
    # Admit only a point-ratio expression, optionally naming the requested
    # output quantity. A multiplier about an input is not an output answer.
    filler = ((option_numbers >= 0) + (w['ratio_factor'][o] > 0) + (w['ratio_option_filler'][o] > 0) + (o == 0)) > 0
    option_grammar = (filler.unsqueeze(-1) + w['ratio_option_target'][o]).amin(-2)
    matching_grammar = (option_grammar * (active > 0).unsqueeze(1)).sum(-1) > 0
    point_valid = numerical_choice_mask(w, o)
    option_units = physical_units(w, o) * (w['skip'][o] == 0).unsqueeze(-1)
    unitless = option_units.sum(dim=(-1, -2)) == 0
    compatible = count_valid * point_valid * unitless * (option_value > 0) * (~ambiguous_option) * matching_grammar
    enough = compatible.sum(-1, keepdim=True) >= 3
    error = (torch.log(option_value.clamp_min(0.000001)).unsqueeze(-1) - torch.log(prediction.clamp_min(0.000001)).unsqueeze(1)).abs()
    evidence = (torch.exp(-w['ratio_constants'][2] * error) * active.unsqueeze(1)).amax(-1)
    magnitude_score = w['ratio_constants'][1] * evidence * compatible * enough * agreement
    # A direction can be certain without a numerical factor. Unknown-magnitude
    # changes are signed constraints, not fictitious unit-sized log ratios.
    residual_change = generic_change * (~explained_change)
    direction = w['ratio_up'][q] - w['ratio_down'][q]
    unknown_valid = residual_change * unambiguous * (nearest > 0) * (direction != 0)
    unknown_count = torch.zeros_like(observations)
    unknown_sign = torch.zeros_like(observations)
    unknown_count.scatter_add_(1, nearest, unknown_valid.to(torch.float32))
    unknown_sign.scatter_add_(1, nearest, direction * unknown_valid)
    unknown_count[:, 0] = 0
    unknown_sign[:, 0] = 0
    all_changed = changed + (unknown_count > 0)
    all_covered = (all_changed.to(torch.float32) @ used.to(torch.float32).t()) == all_changed.sum(-1, keepdim=True)
    repeated_change = ((observations + unknown_count) > 1).sum(-1) > 0
    unattached_change = (residual_change * (~unknown_valid)).sum(-1) > 0
    # A 'two' inside 'distance between two masses' counts the objects; other
    # unbound source numbers are not assumed to describe mere directions.
    between_count = (numbers == 2) * between_marker.roll(1, dims=-1) * distance_marker.roll(2, dims=-1) * embedded_mass.roll(-1, dims=-1) * (position >= 2) * (position + 1 < q.shape[1])
    unbound_source_number = ((numbers >= 0) * source_zone * (ratios <= 0) * (~between_count)).sum(-1) > 0
    qualitative_forbidden = w['qualitative_forbidden'][q].sum(-1) > 0
    qualitative_safe = (~unattached) * (~repeated_change) * (~unattached_change) * (~adjacent_numbers) * (~forbidden) * (~absolute_context) * (~unbound_source_number) * (~qualitative_forbidden)
    applicable = requested * contextual * all_covered * (all_changed.sum(-1, keepdim=True) > 0) * qualitative_safe.unsqueeze(-1)
    applicable = applicable * ((~w['ratio_alignment']) + aligned)
    influence = unknown_sign.unsqueeze(1) * powers.sign().unsqueeze(0)
    possible_up = (influence > 0).sum(-1) > 0
    possible_down = (influence < 0).sum(-1) > 0
    exact_only = (~possible_up) * (~possible_down)
    increase = ((log_prediction > 0.00001) * exact_only) + ((log_prediction >= -0.00001) * possible_up * (~possible_down))
    decrease = ((log_prediction < -0.00001) * exact_only) + ((log_prediction <= 0.00001) * possible_down * (~possible_up))
    unchanged = (log_prediction.abs() <= 0.00001) * exact_only
    certain = increase + decrease + unchanged
    outcome = increase.to(torch.float32) - decrease.to(torch.float32)
    all_certain = ((applicable > 0) * (~certain)).sum(-1, keepdim=True) == 0
    upper_direction = torch.where(applicable > 0, outcome, -2).amax(-1, keepdim=True)
    lower_direction = torch.where(applicable > 0, outcome, 2).amin(-1, keepdim=True)
    direction_agreement = (upper_direction == lower_direction) * all_certain * (applicable.sum(-1, keepdim=True) > 0)
    # Only unquantified direction choices can receive qualitative evidence.
    # 'Increases' does not prove 'quadruples', even when both are upward changes.
    option_up = w['ratio_up'][o].sum(-1) > 0
    option_down = w['ratio_down'][o].sum(-1) > 0
    option_static = w['qualitative_static'][o].sum(-1) > 0
    one_direction = (option_up.to(torch.int64) + option_down.to(torch.int64) + option_static.to(torch.int64)) == 1
    direction_filler = (w['ratio_option_filler'][o] > 0) + (w['ratio_up'][o] > 0) + (w['ratio_down'][o] > 0) + (w['qualitative_static'][o] > 0) + (o == 0)
    direction_grammar = (direction_filler.unsqueeze(-1) + w['ratio_option_target'][o]).amin(-2)
    direction_compatible = (direction_grammar * (applicable > 0).unsqueeze(1)).sum(-1) > 0
    direction_compatible = direction_compatible * one_direction * point_valid * unitless * (numeric_count == 0) * (~ambiguous_option)
    enough_directions = direction_compatible.sum(-1, keepdim=True) >= 2
    option_direction = option_up.to(torch.float32) - option_down.to(torch.float32)
    supported_direction = option_direction == upper_direction
    qualitative_score = w['qualitative_gain'] * supported_direction * direction_compatible * enough_directions * direction_agreement
    return magnitude_score + qualitative_score


def named_trait_purity(w, tokens, entity_index):
    # Check the entire option against the selected entity's declared spellings.
    # Aliases may have different lengths; unknown content remains a barrier.
    ids = w['lex'][tokens]
    position = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    content = (ids != 0) + (w['skip'][tokens] == 0)
    order = torch.where(content > 0, position, position + tokens.shape[1]).argsort(-1)
    sequence = torch.where((ids == 0) * (w['skip'][tokens] == 0), w['base'], ids).gather(1, order)
    names = w['trait_entity_patterns'][entity_index]
    patterns = w['trait_patterns'][names]
    width = w['trait_pattern_width'][names]
    hit = torch.ones((tokens.shape[0], tokens.shape[1], names.shape[-1]), device=tokens.device, dtype=torch.bool)
    for offset in range(4):
        part = sequence.roll(-offset, dims=-1)
        equal = part.unsqueeze(-1) == patterns[:, :, offset].unsqueeze(1)
        valid = (position < tokens.shape[1] - offset).unsqueeze(-1)
        hit = hit * ((equal * valid) + (width <= offset).unsqueeze(1))
    span = (hit * width.unsqueeze(1)).amax(-1)
    covered = span > 0
    for gap in range(1, 4):
        covered = covered + (span.roll(gap, dims=-1) > gap) * (position >= gap)
    grammatical = (w['skip'][tokens] > 0).gather(1, order)
    return (covered + grammatical).amin(-1)


def trait_intersection(w, q, o, focused, focused_bag, affirmed, uncertain):
    # A three-valued property table composes independent requested attributes.
    # Missing facts stay unknown; a strong positive vote needs every constraint.
    matched = (focused_bag @ w['trait_groups'].t() > 0).to(torch.float32)
    matched[:, 0] = 1
    positive_rows = matched[:, w['trait_positive']].amin(-1) * w['trait_has_positive']
    negative_rows = matched[:, w['trait_negative']].amin(-1) * w['trait_has_negative']
    excluded_rows = (matched[:, w['trait_exclusions']] * (w['trait_exclusions'] != 0)).amax(-1)
    unresolved = ((positive_rows + negative_rows) * excluded_rows).sum(-1, keepdim=True) > 0
    attribute = w['trait_attribute'].unsqueeze(0).expand(q.shape[0], -1)
    positive = torch.zeros((q.shape[0], w['trait_cards'].shape[1]), device=q.device)
    negative = torch.zeros_like(positive)
    positive.scatter_add_(1, attribute, positive_rows)
    negative.scatter_add_(1, attribute, negative_rows)
    positive = positive.clamp_max(1)
    negative = negative.clamp_max(1)
    requested = positive - negative
    count = (positive + negative).clamp_max(1).sum(-1, keepdim=True)
    contradictory = (positive * negative).sum(-1, keepdim=True) > 0
    entity_match = affirmed[:, :, w['trait_entity']]
    suppressed = entity_match @ w['trait_dominance'] > 0
    selected = entity_match * (~suppressed)
    one_entity = selected.sum(-1) == 1
    entity_index = selected.argmax(-1)
    entity_property = w['trait_cards'][entity_index]
    # A name plus an unsupported assertion is not a pure entity answer.
    pure_entity = named_trait_purity(w, o.reshape(-1, o.shape[-1]), entity_index.reshape(-1)).reshape(q.shape[0], 4)
    signed = entity_property * requested.unsqueeze(1)
    supported = (signed > 0).sum(-1)
    refuted = (signed < 0).sum(-1)
    complete = (supported == count) * one_entity * pure_entity
    conjunction = w['trait_conjunction'][focused].sum(-1, keepdim=True) > 0
    # A single direct request negator is handled by the shared output polarity.
    # Background, embedded, exceptional ('not only'), and repeated negations
    # remain forbidden instead of negating an incidental property twice.
    direct_scope = w['scope_boundary'][focused].cumsum(-1) == 0
    following_exception = w['option_neg_exception'][focused].roll(-1, dims=-1)
    following_exception[:, -1] = 0
    direct_negator = (w['not_word'][focused] > 0) * direct_scope * (following_exception == 0)
    removable = (w['trait_forbidden'][focused] * direct_negator).sum(-1, keepdim=True)
    removable = removable * (direct_negator.sum(-1, keepdim=True) == 1)
    forbidden = (w['trait_forbidden'][q].sum(-1, keepdim=True) - removable > 0) + (direct_negator.sum(-1, keepdim=True) > 1)
    safe_option = one_entity * pure_entity * (~uncertain) * (w['option_negator'][o].sum(-1) == 0)
    # A closed set of four pure entity options can also express a property
    # without 'and', or coupled labels such as compound / not element / not
    # mixture. At least three choices must have every requested fact known.
    known_property = ((supported + refuted) == count).sum(-1, keepdim=True) >= 3
    closed = (count >= 1) * (safe_option.sum(-1, keepdim=True) == 4) * known_property * (refuted.sum(-1, keepdim=True) > 0)
    composed = (count >= 2) * conjunction
    active = ((closed + composed) > 0) * (~contradictory) * (~forbidden) * (~unresolved) * (complete.sum(-1, keepdim=True) > 0)
    evidence = complete.to(torch.float32) - refuted / count.clamp_min(1)
    return w['trait_gain'] * evidence * safe_option * active


def candidate_choice_mask(w, o):
    # Only this local reader supports affine temperature units. The inherited
    # numerical solvers retain their original unsupported-unit restrictions.
    negated = w['option_negator'][o].sum(-1)
    bounded = w['numeric_bound_word'][o].sum(-1)
    unsupported = (w['numeric_unsupported_unit'][o] * (w['cand_temperature_scale'][o] == 0)).sum(-1)
    unknown = (o == 1).sum(-1)
    return (negated + bounded + unsupported + unknown) == 0


def candidate_readings(w, tokens):
    # Keep the inherited reader unchanged. This local extension recognizes
    # explicit speed units and separately bound affine temperature readings.
    readings, valid, ambiguous = unit_readings(w, tokens)
    units = physical_units(w, tokens)
    numbers = w['number'][tokens]
    position = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    preceding = torch.full_like(numbers, -1)
    bridge = w['unit_bridge'][tokens] > 0
    for gap in range(1, 4):
        candidate = numbers.roll(gap, dims=1)
        intervening = torch.ones_like(bridge)
        for inner in range(gap - 1):
            intervening = intervening * bridge.roll(inner + 1, dims=1)
        linked = (preceding < 0) * (candidate >= 0) * (position >= gap) * intervening
        preceding = torch.where(linked, candidate, preceding)
    temperature_preceding = torch.full_like(numbers, -1)
    temperature_bridge = w['cand_temperature_bridge'][tokens] > 0
    for gap in range(1, 4):
        candidate = numbers.roll(gap, dims=1)
        intervening = torch.ones_like(temperature_bridge)
        for inner in range(gap - 1):
            intervening = intervening * temperature_bridge.roll(inner + 1, dims=1)
        linked = (temperature_preceding < 0) * (candidate >= 0) * (position >= gap) * intervening
        temperature_preceding = torch.where(linked, candidate, temperature_preceding)
    temperature_scale = w['cand_temperature_scale'][tokens]
    temperature_valid = (temperature_scale > 0) * (temperature_preceding >= 0)
    temperature_reading = (temperature_preceding.clamp_min(0) * temperature_scale + w['cand_temperature_offset'][tokens]) * temperature_valid
    temperature_orphan = ((temperature_scale > 0) * (~temperature_valid)).sum(-1) > 0
    temperature_count = temperature_valid.sum(-1, keepdim=True)
    temperature_value = temperature_reading.sum(-1, keepdim=True) / temperature_count.clamp_min(1)
    pair = torch.zeros((tokens.shape[0], tokens.shape[1], 3), device=tokens.device)
    denominator = torch.zeros_like(numbers)
    speed_bridge = w['cand_speed_bridge'][tokens] > 0
    for gap in range(1, 4):
        following = units[:, :, 3].roll(-gap, dims=1)
        intervening = torch.ones_like(speed_bridge)
        for inner in range(gap - 1):
            intervening = intervening * speed_bridge.roll(-inner - 1, dims=1)
        paired = (units[:, :, 2] > 0) * (following > 0) * intervening * (position < tokens.shape[1] - gap)
        pair[:, :, gap - 1] = paired
        denominator = denominator + following * paired
    compound = (pair.sum(-1) == 1) * (preceding >= 0)
    named = w['cand_speed_units'][tokens]
    speed_valid = compound + ((named > 0) * (preceding >= 0))
    speed_scale = units[:, :, 2] / denominator.clamp_min(0.000001) * compound + named
    speed_reading = speed_scale * preceding.clamp_min(0) * speed_valid
    # A compound speed is not also an independent length measurement.
    readings = torch.cat((readings[:, :, :2], readings[:, :, 2:3] * (~compound).unsqueeze(-1), readings[:, :, 3:], speed_reading.unsqueeze(-1)), dim=-1)
    valid = torch.cat((valid[:, :, :2], valid[:, :, 2:3] * (~compound).unsqueeze(-1), valid[:, :, 3:], speed_valid.unsqueeze(-1)), dim=-1)
    consumed_tail = torch.zeros_like(numbers)
    for gap in range(1, 4):
        consumed_tail = consumed_tail + (compound * pair[:, :, gap - 1]).roll(gap, dims=1)
    unit_present = (units.sum(-1) > 0) * (((w['skip'][tokens] == 0) + (preceding >= 0)) > 0)
    assigned = (valid.sum(-1) > 0) + (consumed_tail > 0)
    orphan = (unit_present * (~assigned)).sum(-1) > 0
    counts = valid.sum(1)
    values = readings.sum(1) / counts.clamp_min(1)
    values = torch.cat((values, torch.ones((tokens.shape[0], 1), device=tokens.device), temperature_value), dim=-1)
    known = torch.cat((counts == 1, torch.ones((tokens.shape[0], 1), device=tokens.device, dtype=torch.bool), temperature_count == 1), dim=-1)
    return values, known, valid.sum(dim=(-1, -2)) + temperature_count.squeeze(-1), ambiguous + orphan + temperature_orphan


def kinetic_energy_arithmetic(w, q, o, focused, focused_bag):
    # Static massive-body KE = (1/2) m v^2. Grams convert to kilograms.
    position = torch.arange(q.shape[1], device=q.device).unsqueeze(0)
    target_first = torch.where(focused == w['ke_keyword'], position, q.shape[1]).amin(-1)
    other_first = torch.where(w['ke_other_target'][focused] > 0, position, q.shape[1]).amin(-1)
    request = (focused_bag @ w['ke_query'] > 0) * (target_first < other_first)
    previous = q.roll(1, dims=-1)
    previous[:, 0] = 0
    # 'in joules' is an output-unit request, not an orphan input measurement.
    output_unit = (previous == w['ke_in_word']) * (focused > 0) * (w['ke_output_scale'][q] > 0)
    inputs = torch.where(output_unit, torch.zeros_like(q), q)
    values, known, count, ambiguous = candidate_readings(w, inputs)
    numeric_count = (w['number'][q] >= 0).sum(-1)
    forbidden = (w['ke_forbidden'][q].sum(-1) + w['option_negator'][q].sum(-1) + w['numeric_unsupported_unit'][q].sum(-1)) > 0
    output_count = output_unit.sum(-1)
    requested_scale = (w['ke_output_scale'][q] * output_unit).amax(-1)
    default_scale = torch.where(output_count == 0, torch.ones_like(requested_scale), requested_scale)
    active = request * known[:, 0] * known[:, 11] * (values[:, 0] > 0) * (values[:, 11] >= 0)
    active = active * (count == 2) * (numeric_count == 2) * (output_count <= 1) * (~ambiguous) * (~forbidden)
    prediction = w['ke_coefficient'] * values[:, 0] * values[:, 11].square()
    numbers = w['number'][o]
    count_number = (numbers >= 0).sum(-1)
    value = numbers.amax(-1).clamp_min(0)
    scales = w['ke_output_scale'][o]
    unit_count = (scales > 0).sum(-1)
    scale = torch.where(unit_count == 0, default_scale.unsqueeze(-1), scales.amax(-1))
    pure = (((numbers >= 0) + (scales > 0) + (w['ke_output_words'][o] > 0) + (w['skip'][o] > 0)) > 0).amin(-1)
    compatible = (count_number == 1) * (unit_count <= 1) * pure
    enough = compatible.sum(-1, keepdim=True) == 4
    observed = value * scale
    error = (observed - prediction.unsqueeze(-1)).abs() / (prediction.abs().unsqueeze(-1) + 1)
    return w['knobs'][8] * torch.exp(-w['knobs'][9] * error) * active.unsqueeze(-1) * compatible * enough


def candidate_numerical(w, q, o, focused, focused_bag):
    # Evaluate laws inside candidate descriptions, rather than treating each
    # number as a proposed answer to an arithmetic problem in the question.
    tokens = o.reshape(-1, o.shape[-1])
    values, known, reading_count, ambiguous = candidate_readings(w, tokens)
    numbers = w['number'][tokens]
    fully_bound = ((numbers >= 0).sum(-1, keepdim=True) == w['cand_measurement_count']) * (reading_count.unsqueeze(-1) == w['cand_measurement_count'])
    left = values[:, w['cand_left']]
    right = values[:, w['cand_right']]
    domain = ((left > 0) + ((left == 0) * w['cand_left_zero'])) * ((right > 0) + ((right == 0) * w['cand_right_zero']))
    known_law = known[:, w['cand_left']] * known[:, w['cand_right']] * domain * fully_bound * (~ambiguous).unsqueeze(-1)
    forbidden = (w['cand_forbidden'][q].sum(1) + w['cand_forbidden'][o].sum(dim=(1, 2))) > 0
    # Replace only invalid divisors before evaluating all templates. Preserve
    # actual zero factors in products and squares instead of clamping them up.
    divisor = torch.where((right == 0) * (w['cand_exponent'] < 0), torch.ones_like(right), right)
    magnitude = left * divisor.pow(w['cand_exponent'])
    magnitude = magnitude.reshape(q.shape[0], 4, -1)
    known_law = known_law.reshape(q.shape[0], 4, -1).amin(1)
    request = (focused_bag @ w['cand_query'].t() > 0) * known_law * (~forbidden)
    likely_next = w['cmp_likely'][focused].roll(-1, dims=-1)
    likely_next[:, -1] = 0
    up = ((w['cmp_up'][focused] > 0) * (likely_next == 0)).sum(-1) > 0
    down = ((w['cmp_down'][focused] > 0) * (likely_next == 0)).sum(-1) > 0
    direction = up.to(torch.float32) - down.to(torch.float32)
    ordered = magnitude * direction.reshape(-1, 1, 1)
    # Float32 unit conversions must not break physically equal candidate ties.
    # A machine-precision bound is scale-relative, not fitted to examples.
    tolerance = w['cand_roundoff'] * magnitude.abs().amax(1, keepdim=True).clamp_min(w['cand_tiny'])
    highest = ordered.amax(1, keepdim=True)
    lowest = ordered.amin(1, keepdim=True)
    extreme = ((highest - ordered) <= tolerance) * ((highest - lowest) > tolerance)
    support = (extreme * request.unsqueeze(1)).sum(-1) > 0
    safe = (candidate_choice_mask(w, o).sum(-1, keepdim=True) == 4) * (w['cand_scope_forbidden'][q].sum(-1, keepdim=True) == 0)
    active = (request.sum(-1, keepdim=True) == 1) * (direction != 0).unsqueeze(-1) * safe
    return w['cand_gain'] * support * active


def candidate_comparisons(w, q, o, focused, whole, ob):
    # Finite, signed causal axes define a partial order over stated conditions.
    # Do not add arbitrary magnitudes: a winner must dominate every opponent.
    qm = (whole @ w['cmp_groups'].t() > 0).to(torch.float32)
    qm[:, 0] = 1
    topic = qm[:, w['cmp_required']].amin(-1)
    excluded = (qm[:, w['cmp_excluded']] * (w['cmp_excluded'] != 0)).amax(-1)
    om = (ob @ w['cmp_groups'].t() > 0).to(torch.float32)
    om[:, :, 0] = 1
    # Applicability restrictions also apply inside candidate descriptions.
    option_excluded = (om[:, :, w['cmp_excluded']] * (w['cmp_excluded'] != 0)).amax(-1).amax(1)
    excluded = torch.maximum(excluded, option_excluded)
    high = om[:, :, w['cmp_positive']].amin(-1)
    low = om[:, :, w['cmp_negative']].amin(-1)
    state = high - low
    stated = (high + low == 1).sum(1)
    complete = stated == 4
    partial = ((high + low).sum(1) > 0) * (~complete)
    usable = complete.to(torch.float32).unsqueeze(-1) * w['cmp_axis_domain']
    incomplete = partial.to(torch.float32) @ w['cmp_axis_domain']
    likely_next = w['cmp_likely'][focused].roll(-1, dims=-1)
    likely_next[:, -1] = 0
    up = ((w['cmp_up'][focused] > 0) * (likely_next == 0)).sum(-1) > 0
    down = ((w['cmp_down'][focused] > 0) * (likely_next == 0)).sum(-1) > 0
    direction = up.to(torch.float32) - down.to(torch.float32)
    difference = (state.unsqueeze(2) - state.unsqueeze(1)) * direction.reshape(-1, 1, 1, 1)
    better = torch.einsum('nija,nad->nijd', (difference > 0).to(torch.float32), usable) > 0
    worse = torch.einsum('nija,nad->nijd', (difference < 0).to(torch.float32), usable) > 0
    winner = (better * (~worse)).sum(2) == 3
    safe_q = w['cmp_forbidden'][q].sum(-1, keepdim=True) == 0
    safe_o = w['option_negator'][o].sum(dim=(-1, -2), keepdim=False) == 0
    active = topic * (1 - excluded) * (incomplete == 0) * (usable.sum(1) > 0) * safe_q * safe_o.unsqueeze(-1)
    supported = (winner * active.unsqueeze(1)).sum(-1) > 0
    agreement = supported.sum(-1, keepdim=True) == 1
    return w['cmp_gain'] * supported * agreement


def pathway_matches(w, tokens):
    # Keep grammatical skipping and unknown-word barriers identical to bag().
    ids = w['lex'][tokens]
    position = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    content = (ids != 0) + (w['skip'][tokens] == 0)
    order = torch.where(content > 0, position, position + tokens.shape[1]).argsort(-1)
    sequence = torch.where((ids == 0) * (w['skip'][tokens] == 0), w['base'], ids).gather(1, order)
    hit = torch.ones((tokens.shape[0], tokens.shape[1], w['order_patterns'].shape[0]), device=tokens.device, dtype=torch.bool)
    for offset in range(4):
        needed = w['order_width'] > offset
        part = sequence.roll(-offset, dims=-1)
        equal = part.unsqueeze(-1) == w['order_patterns'][:, offset]
        valid = (position < tokens.shape[1] - offset).unsqueeze(-1)
        hit = hit * ((equal * valid) + (~needed))
    hit = hit * (w['order_width'] > 0)
    # Longest phrase wins, so 'organ system' cannot also count as 'organ'.
    span = (hit * w['order_width']).amax(-1)
    hit = hit * (w['order_width'] == span.unsqueeze(-1))
    covered = torch.zeros_like(span, dtype=torch.bool)
    for gap in range(1, 4):
        covered = covered + (span.roll(gap, dims=-1) > gap) * (position >= gap)
    hit = hit * (~covered).unsqueeze(-1)
    return hit, order


def pathway_mentions(w, tokens):
    hit, token_order = pathway_matches(w, tokens)
    position = torch.arange(tokens.shape[1], device=tokens.device).reshape(1, -1, 1)
    first = torch.where(hit, position, tokens.shape[1]).amin(1)
    last = torch.where(hit, position, -1).amax(1)
    return first, last


def ordered_pathways(w, q, o, whole, focused_bag, options_bag):
    # Validate relative positions, never a bag-of-words ordering heuristic.
    question = (whole @ w['order_groups'].t() > 0)
    question[:, 0] = True
    shared = (options_bag.amin(1) @ w['order_groups'].t() > 0)
    context = question + shared
    # Ordering instructions can precede the final which/what selection.
    request = (whole @ w['order_groups'].t() > 0)
    request[:, 0] = True
    # The question must name the topic. Additional context may be supplied by
    # entities common to all four candidate lists (e.g. biological levels).
    relevant = context[:, w['order_required']].amin(-1) * question[:, w['order_required'][:, 0]]
    upward = request[:, w['order_up']].amin(-1)
    downward = request[:, w['order_down']].amin(-1)
    direction = upward.to(torch.float32) - downward.to(torch.float32)
    direction = torch.where((upward + downward) == 0, w['order_default'].to(torch.float32), direction)
    option_tokens = o.reshape(-1, o.shape[-1])
    hit, token_order = pathway_matches(w, option_tokens)
    position = torch.arange(o.shape[-1], device=o.device).reshape(1, -1, 1)
    first = torch.where(hit, position, o.shape[-1]).amin(1)
    last = torch.where(hit, position, -1).amax(1)
    # Per-template coverage rejects appended assertions and unrecognized nodes,
    # rather than interpreting their accidental noun order as a valid list.
    span = (hit * w['order_width']).to(torch.float32) @ w['order_incidence']
    covered = span > 0
    for gap in range(1, 4):
        covered = covered + (span.roll(gap, dims=1) > gap) * (position >= gap)
    filler = ((w['skip'][option_tokens] + w['order_filler'][option_tokens]) > 0).gather(1, token_order)
    pure = (covered + filler.unsqueeze(-1)).amin(1).reshape(q.shape[0], 4, -1)
    node_first = first[:, w['order_nodes']].amin(-1).reshape(q.shape[0], 4, w['order_nodes'].shape[0], -1)
    node_last = last[:, w['order_nodes']].amax(-1).reshape_as(node_first)
    present = node_first < o.shape[-1]
    unique = ((node_first == node_last) + (~present)).amin(-1)
    count = present.sum(-1)
    same_nodes = (present == present[:, :1]).amin(dim=(1, 3))
    rank = torch.arange(node_first.shape[-1], device=q.device)
    pairs = (rank.unsqueeze(0) > rank.unsqueeze(1)) * present.unsqueeze(-1) * present.unsqueeze(-2)
    difference = node_first.unsqueeze(-2) - node_first.unsqueeze(-1)
    correct = (difference * direction.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)) > 0
    monotone = (correct + (~pairs)).amin(dim=(-1, -2))
    all_candidates = ((count >= w['order_threshold']) * unique * pure).amin(1) * same_nodes
    order_request = (focused_bag @ w['order_query'] > 0).unsqueeze(-1) + upward + downward
    active = relevant * (direction != 0) * all_candidates * order_request
    # Respect an explicit starting stage, even for a cyclic pathway. A default
    # rotation must not overrule 'beginning with pupa' or an unknown anchor.
    qposition = torch.arange(q.shape[1], device=q.device).unsqueeze(0)
    link = w['order_anchor_link'][q].roll(-1, dims=-1)
    link[:, -1] = 0
    anchor_words = w['order_anchor'][q] > 0
    cue = anchor_words * (link > 0)
    cue_count = anchor_words.sum(-1)
    cue_position = (cue * qposition).amax(-1)
    qhit, qorder = pathway_matches(w, q)
    after = (qorder > cue_position.unsqueeze(-1) + 1) * (qorder <= cue_position.unsqueeze(-1) + 6)
    anchor_position = torch.where(qhit * after.unsqueeze(-1), qorder.unsqueeze(-1), q.shape[1]).amin(1)
    nearest = anchor_position.amin(-1)
    selected_anchor = (anchor_position == nearest.unsqueeze(-1)) * (nearest < q.shape[1]).unsqueeze(-1)
    anchor_nodes = selected_anchor[:, w['order_nodes']].amax(-1)
    one_anchor = anchor_nodes.sum(-1) == 1
    bridge = (w['skip'][q] == 0) * (qposition > cue_position.unsqueeze(-1)) * (qposition < nearest.unsqueeze(-1))
    clear_anchor = (cue_count == 1) * (cue.sum(-1) == 1) * (nearest < q.shape[1]) * (bridge.sum(-1) == 0)
    earliest = (node_first == node_first.amin(-1, keepdim=True)) * present
    starts_correct = (earliest * anchor_nodes.unsqueeze(1)).sum(-1) > 0
    anchor_ok = (cue_count == 0).reshape(-1, 1, 1) + starts_correct * one_anchor.unsqueeze(1) * clear_anchor.reshape(-1, 1, 1)
    supported = (monotone * anchor_ok * active.unsqueeze(1)).sum(-1) > 0
    # A unique supported candidate is required across every applicable pathway.
    consensus = supported.sum(-1, keepdim=True) == 1
    safe = (w['option_negator'][o] + w['order_forbidden'][o]).sum(dim=(-1, -2)) == 0
    # A least-likely request is not a hierarchy's least-to-most direction.
    uncertain_request = (w['least'][q].sum(-1) > 0) * (w['cmp_likely'][q].sum(-1) > 0)
    return w['order_gain'] * supported * consensus * (safe * (~uncertain_request)).unsqueeze(-1)


def force_directions(w, tokens, anchors):
    # Bind a direction through at most five declared bridge tokens, never
    # through a conjunction. Equidistant disagreeing labels are ambiguous.
    position = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    code = w['force_axis'][tokens] * w['force_sign'][tokens].to(torch.int64)
    bridge = (w['force_bridge'][tokens] > 0) + (w['number'][tokens] >= 0)
    chosen = torch.zeros_like(tokens)
    chosen_position = torch.zeros_like(tokens)
    found = torch.zeros_like(tokens, dtype=torch.bool)
    conflict = torch.zeros_like(found)
    for gap in range(1, 6):
        before_bridge = torch.ones_like(found)
        after_bridge = torch.ones_like(found)
        for inner in range(gap - 1):
            before_bridge = before_bridge * bridge.roll(inner + 1, dims=-1)
            after_bridge = after_bridge * bridge.roll(-inner - 1, dims=-1)
        before = code.roll(gap, dims=-1)
        after = code.roll(-gap, dims=-1)
        before_valid = (before != 0) * before_bridge * (position >= gap)
        after_valid = (after != 0) * after_bridge * (position + gap < tokens.shape[1])
        hit = (before_valid + after_valid) * (~found)
        conflict = conflict + hit * before_valid * after_valid * (before != after)
        chosen = torch.where(hit, torch.where(after_valid, after, before), chosen)
        location = torch.where(after_valid, position + gap, position - gap)
        chosen_position = torch.where(hit, location, chosen_position)
        found = found + hit
    valid = found * (~conflict) * anchors
    return chosen.abs() * valid, chosen.sign().to(torch.float32) * valid, valid, chosen_position


def collinear_forces(w, q, o, focused_bag):
    readings, measured, ambiguous = unit_readings(w, q)
    force = measured[:, :, 4]
    axis, sign, direction_known, label_position = force_directions(w, q, force)
    label_usage = torch.zeros_like(q)
    label_usage.scatter_add_(1, label_position.clamp(0, q.shape[1] - 1), direction_known.to(torch.int64))
    unique_binding = label_usage.amax(-1) <= 1
    count = force.sum(-1)
    common_axis = axis.amax(-1, keepdim=True)
    aligned = ((axis == common_axis) + (~force)).amin(-1)
    complete = (direction_known.sum(-1) == count) * (common_axis[:, 0] > 0)
    complete = complete * ((w['force_axis'][q] > 0).sum(-1) == count)
    numeric_count = (w['number'][q] >= 0).sum(-1)
    following_count_noun = w['force_count_noun'][q].roll(-1, dims=-1)
    following_count_noun[:, -1] = 0
    count_marker = (w['number'][q] == count.unsqueeze(-1)) * (following_count_noun > 0)
    numeric_count = numeric_count - count_marker.sum(-1)
    only_force_units = (measured.sum(dim=(-1, -2)) == count) * (count_marker.sum(-1) <= 1)
    net = (readings[:, :, 4] * sign).sum(-1, keepdim=True)
    magnitude = net.abs()
    flat = o.reshape(-1, o.shape[-1])
    numbers = w['number'][flat]
    numeric = numbers >= 0
    option_axis, option_sign, option_known, option_label_position = force_directions(w, flat, numeric)
    label_count = (w['force_axis'][flat] > 0).sum(-1)
    labelled = (label_count == 1) * (option_known.sum(-1) == 1)
    axis_value = option_axis.sum(-1).reshape(q.shape[0], 4)
    sign_value = option_sign.sum(-1).reshape(q.shape[0], 4)
    labelled = labelled.reshape(q.shape[0], 4)
    label_count = label_count.reshape(q.shape[0], 4)
    grammar = numeric + (w['force_filler'][flat] > 0) + (w['force_axis'][flat] > 0) + (flat == 0)
    pure = (grammar > 0).amin(-1).reshape(q.shape[0], 4)
    point = (numeric.sum(-1) == 1).reshape(q.shape[0], 4)
    value = numbers.amax(-1).clamp_min(0).reshape(q.shape[0], 4)
    same_format = (label_count.sum(-1) == 0) + (labelled.sum(-1) == 4)
    label_ok = (label_count == 0) + labelled * (axis_value == common_axis) * (sign_value == net.sign()) * (magnitude > 0)
    # Standalone "up" is a valid direction, unlike the bound phrase "up to".
    # The generic point-choice mask treats every "up" as an inequality cue.
    following = o.roll(-1, dims=-1)
    following[:, :, -1] = 0
    up_to = (o == w['numeric_bound_up_to'][0]) * (following == w['numeric_bound_up_to'][1])
    bounded = ((w['numeric_bound_word'][o] > 0) * (w['force_axis'][o] == 0)) + up_to
    unsafe = w['option_negator'][o] + w['numeric_unsupported_unit'][o]
    safe_point = (bounded.sum(-1) == 0) * (unsafe.sum(-1) == 0)
    compatible = pure * point * safe_point
    matches = ((value - magnitude).abs() <= 0.0001 * (1 + magnitude)) * label_ok
    request = focused_bag @ w['force_query'] > 0
    grammar = (w['number'][q] >= 0) + (w['force_question_filler'][q] > 0) + (w['force_axis'][q] > 0) + (q == 0)
    pure_question = (grammar > 0).amin(-1)
    safe_question = pure_question * (w['force_forbidden'][q].sum(-1) == 0) * numerical_question_mask(w, q)
    active = request * safe_question * aligned * complete * unique_binding * (~ambiguous)
    active = active * (count >= 2) * (count <= 3) * (numeric_count == count) * only_force_units
    active = active * (compatible.sum(-1) == 4) * same_format * (matches.sum(-1) == 1)
    return w['knobs'][8] * matches * active.unsqueeze(-1)


def conversion_units(w, tokens):
    # Exact raw-token phrases: never join over an unknown word or a clause.
    position = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    code = torch.zeros_like(tokens)
    unit = torch.zeros_like(tokens)
    span = torch.zeros_like(tokens)
    for width in range(1, 4):
        code = code * w['convert_base'] + tokens.roll(1 - width, dims=-1)
        index = torch.searchsorted(w['convert_codes'], code.contiguous())
        found = (w['convert_codes'][index] == code) * (position + width <= tokens.shape[1])
        unit = torch.where(found, w['convert_ids'][index], unit)
        span = torch.where(found, width, span)
    inside = torch.zeros_like(tokens, dtype=torch.bool)
    for gap in range(1, 3):
        inside = inside + (span.roll(gap, dims=-1) > gap) * (position >= gap)
    unit = unit * (~inside)
    span = span * (~inside)
    covered = span > 0
    for gap in range(1, 3):
        covered = covered + (span.roll(gap, dims=-1) > gap) * (position >= gap)
    return unit, covered


def direct_unit_conversion(w, q, o):
    qunit, qcovered = conversion_units(w, q)
    numbers = w['convert_number'][q]
    numeric = numbers >= 0
    preceding = numbers.roll(1, dims=-1)
    preceding[:, 0] = -1
    article = w['convert_article'][q].roll(1, dims=-1) > 0
    article[:, 0] = 0
    dimension = w['convert_dimension'][qunit]
    explicit = (qunit > 0) * (preceding >= 0)
    implicit = (qunit > 0) * article * (dimension != w['convert_temperature_dimension'])
    source = explicit + implicit
    source_id = (qunit * source).sum(-1, keepdim=True)
    # Clamp invalid multiple-source indices before lookup; the active mask
    # rejects those inputs independently of the numerical calculation.
    source_id = source_id.clamp(0, w['convert_scale'].shape[0] - 1)
    source_value = (torch.where(explicit, preceding, torch.ones_like(preceding)) * source).sum(-1, keepdim=True)
    source_dimension = w['convert_dimension'][source_id]
    base_value = source_value * w['convert_scale'][source_id] + w['convert_offset'][source_id]
    target = (qunit > 0) * (~source)
    target_count = target.sum(-1, keepdim=True)
    target_id = (qunit * target).sum(-1, keepdim=True).clamp(0, w['convert_scale'].shape[0] - 1)
    has_target = target_count == 1
    explicit_count = explicit.sum(-1)
    implicit_count = implicit.sum(-1)
    one_source = (source.sum(-1) == 1) * (numeric.sum(-1) == explicit_count)
    one_source = one_source * ((explicit_count == 1) + (implicit_count == 1))
    question_pure = ((qcovered + numeric + (w['convert_question_grammar'][q] > 0) + (q == 0)) > 0).amin(-1)
    safe_question = question_pure * (w['convert_forbidden'][q].sum(-1) == 0)
    equivalence = w['convert_equivalence'][q].sum(-1, keepdim=True) > 0
    request = (target_count <= 1) * (has_target + equivalence)

    flat = o.reshape(-1, o.shape[-1])
    ounit, ocovered = conversion_units(w, flat)
    onumbers = w['convert_number'][flat]
    onumeric = onumbers >= 0
    oprevious = onumbers.roll(1, dims=-1)
    oprevious[:, 0] = -1
    bound = (ounit > 0) * (oprevious >= 0)
    unit_count = (ounit > 0).sum(-1).reshape(q.shape[0], 4)
    option_id = ounit.sum(-1).reshape(q.shape[0], 4).clamp(0, w['convert_scale'].shape[0] - 1)
    chosen_id = torch.where(has_target, target_id, option_id)
    has_option_unit = unit_count == 1
    unit_format = ((unit_count == 0) * has_target) + has_option_unit * (bound.sum(-1).reshape(q.shape[0], 4) == 1)
    target_match = (~has_target) + (unit_count == 0) + (option_id == target_id)
    compatible = w['convert_dimension'][chosen_id] == source_dimension
    point = (onumeric.sum(-1) == 1).reshape(q.shape[0], 4)
    option_pure = ((ocovered + onumeric + (w['convert_option_grammar'][flat] > 0) + (flat == 0)) > 0).amin(-1).reshape(q.shape[0], 4)
    valid_options = unit_format * target_match * compatible * point * option_pure
    option_value = onumbers.amax(-1).clamp_min(0).reshape(q.shape[0], 4)
    scale = w['convert_scale'][chosen_id]
    offset = w['convert_offset'][chosen_id]
    prediction = (base_value - offset) / scale
    tolerance = w['convert_roundoff'] * (1 + prediction.abs() + base_value.abs() / scale + offset.abs() / scale)
    temperature = source_dimension == w['convert_temperature_dimension']
    exact = w['convert_exact'][q].sum(-1, keepdim=True) > 0
    tolerance = tolerance + 0.5 * temperature * (~exact)
    matches = (option_value - prediction).abs() <= tolerance
    physical = (~temperature) + (base_value >= 0)
    active = one_source.unsqueeze(-1) * safe_question.unsqueeze(-1) * request * physical
    active = active * (valid_options.sum(-1, keepdim=True) == 4) * (matches.sum(-1, keepdim=True) == 1)
    return w['knobs'][8] * matches.to(torch.float32) * active


def entity_name_spans(w, tokens):
    # Longest names suppress subnames only at overlapping occurrences.
    # 'animals, birds, reptiles' must retain animals as an independent entity.
    ids = w['lex'][tokens]
    position = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    content = (ids != 0) + (w['skip'][tokens] == 0)
    order = torch.where(content > 0, position, position + tokens.shape[1]).argsort(-1)
    sequence = torch.where((ids == 0) * (w['skip'][tokens] == 0), w['base'], ids).gather(1, order)
    hit = torch.ones((tokens.shape[0], tokens.shape[1], w['trait_patterns'].shape[0]), device=tokens.device, dtype=torch.bool)
    for offset in range(4):
        needed = w['trait_pattern_width'] > offset
        part = sequence.roll(-offset, dims=-1)
        equal = part.unsqueeze(-1) == w['trait_patterns'][:, offset]
        valid = (position < tokens.shape[1] - offset).unsqueeze(-1)
        hit = hit * ((equal * valid) + (~needed))
    hit = hit * (w['trait_pattern_width'] > 0)
    span = (hit * w['trait_pattern_width']).amax(-1)
    hit = hit * (w['trait_pattern_width'] == span.unsqueeze(-1))
    inside = torch.zeros_like(span, dtype=torch.bool)
    for gap in range(1, 4):
        inside = inside + (span.roll(gap, dims=-1) > gap) * (position >= gap)
    hit = hit * (~inside).unsqueeze(-1)
    span = (hit * w['trait_pattern_width']).amax(-1)
    covered = span > 0
    for gap in range(1, 4):
        covered = covered + (span.roll(gap, dims=-1) > gap) * (position >= gap)
    entity = (hit.amax(1).to(torch.float32) @ w['shared_pattern_entities'] > 0).to(torch.float32)
    raw_coverage = covered.gather(1, order.argsort(-1))
    occurrences = hit.amax(-1).sum(-1)
    return entity, raw_coverage, occurrences


def shared_entity_mentions(w, tokens):
    entity, covered, occurrences = entity_name_spans(w, tokens)
    grammatical = (w['skip'][tokens] > 0) + (w['shared_question_filler'][tokens] > 0)
    pure = ((covered + grammatical) > 0).amin(-1)
    return entity, pure


def common_property_evidence(w, q, o, whole, ob):
    # Reverse property inference: a proposed common property must be known
    # true of EVERY named question entity. Missing facts do not vote true.
    selected, pure_question = shared_entity_mentions(w, q)
    entity_count = selected.sum(-1, keepdim=True)
    shared_positive = (selected @ (w['trait_cards'] > 0).to(torch.float32)) == entity_count
    shared_negative = (selected @ (w['trait_cards'] < 0).to(torch.float32)) == entity_count
    flat = ob.reshape(-1, ob.shape[-1])
    matched = (flat @ w['trait_groups'].t() > 0).to(torch.float32)
    matched[:, 0] = 1
    excluded = (matched[:, w['trait_exclusions']] * (w['trait_exclusions'] != 0)).amax(-1)
    positive_rows = matched[:, w['trait_positive']].amin(-1) * w['trait_has_positive'] * (1 - excluded)
    negative_rows = matched[:, w['trait_negative']].amin(-1) * w['trait_has_negative'] * (1 - excluded)
    attribute = w['trait_attribute'].unsqueeze(0).expand(flat.shape[0], -1)
    positive = torch.zeros((flat.shape[0], w['trait_cards'].shape[1]), device=q.device)
    negative = torch.zeros_like(positive)
    positive.scatter_add_(1, attribute, positive_rows)
    negative.scatter_add_(1, attribute, negative_rows)
    positive = positive.clamp_max(1).reshape(q.shape[0], 4, -1)
    negative = negative.clamp_max(1).reshape(q.shape[0], 4, -1)
    count = (positive + negative).sum(-1)
    contradictory = (positive * negative).sum(-1) > 0
    support = (positive * shared_positive.unsqueeze(1) + negative * shared_negative.unsqueeze(1)).sum(-1)
    counterexample_positive = selected @ (w['trait_cards'] < 0).to(torch.float32) > 0
    counterexample_negative = selected @ (w['trait_cards'] > 0).to(torch.float32) > 0
    refuted = (positive * counterexample_positive.unsqueeze(1) + negative * counterexample_negative.unsqueeze(1)).sum(-1)
    valid_frames = torch.sparse.mm(w['shared_frames'], torch.cat((positive_rows, negative_rows), dim=-1).t()).t() > 0
    coverage = torch.sparse.mm(w['shared_coverage'], (flat * valid_frames).t()).t()
    tokens = o.reshape(-1, o.shape[-1])
    covered = coverage.gather(1, w['lex'][tokens]) > 0
    grammatical = (w['skip'][tokens] > 0) + (w['shared_filler'][tokens] > 0)
    pure = ((covered + grammatical) > 0).amin(-1).reshape(q.shape[0], 4)
    # Do not infer truth of alternatives, comparisons, conditions, or an
    # unresolved negative clause by recognizing one incidental property.
    unsafe_option = (w['option_negator'][o] + w['shared_forbidden'][o]).sum(-1) > 0
    readable = pure * (~unsafe_option) * (~contradictory) * (count > 0)
    supported = (support == count) * readable
    cue = whole @ w['shared_cue'] > 0
    safe_question = (w['shared_forbidden'][q].sum(-1) == 0) * pure_question
    active = cue * safe_question * (entity_count[:, 0] >= 2) * (entity_count[:, 0] <= 4)
    active = active * (supported.sum(-1) == 1) * (readable.sum(-1) >= 2)
    evidence = supported.to(torch.float32) - refuted / count.clamp_min(1) * readable
    return w['trait_gain'] * evidence * active.unsqueeze(-1)


def grouped_property_evidence(w, q, o, whole):
    # A universal list answer must satisfy EVERY requested property for EVERY
    # named member. A mixed pair cannot borrow one property from each member.
    matched = (whole @ w['trait_groups'].t() > 0).to(torch.float32)
    matched[:, 0] = 1
    positive_rows = matched[:, w['trait_positive']].amin(-1) * w['trait_has_positive']
    negative_rows = matched[:, w['trait_negative']].amin(-1) * w['trait_has_negative']
    excluded = (matched[:, w['trait_exclusions']] * (w['trait_exclusions'] != 0)).amax(-1)
    unresolved = ((positive_rows + negative_rows) * excluded).sum(-1) > 0
    attribute = w['trait_attribute'].unsqueeze(0).expand(q.shape[0], -1)
    positive = torch.zeros((q.shape[0], w['trait_cards'].shape[1]), device=q.device)
    negative = torch.zeros_like(positive)
    positive.scatter_add_(1, attribute, positive_rows)
    negative.scatter_add_(1, attribute, negative_rows)
    positive = positive.clamp_max(1)
    negative = negative.clamp_max(1)
    count = (positive + negative).sum(-1, keepdim=True)
    contradictory = (positive * negative).sum(-1) > 0
    valid_frames = torch.sparse.mm(w['shared_frames'], torch.cat((positive_rows, negative_rows), dim=-1).t()).t() > 0
    coverage = torch.sparse.mm(w['shared_coverage'], (whole * valid_frames).t()).t()
    covered = coverage.gather(1, w['lex'][q]) > 0
    question_pure = ((covered + (w['group_question_filler'][q] > 0) + (q == 0)) > 0).amin(-1)
    request_noun = ((w['group_nouns'][q] > 0) * (~covered)).sum(-1) > 0
    universal = ((w['group_universal'][q] > 0) * (~covered)).sum(-1) > 0
    pair = ((w['group_pair'][q] > 0) * (~covered)).sum(-1, keepdim=True) > 0
    # "Only A and B" can denote a union of categories, not a conjunction.
    # Its scope is outside this universal-conjunction reader.
    only = w['group_only'][q] > 0
    only_count = only.sum(-1)
    union_scope = (only_count > 0) * (w['group_and'][q].sum(-1) > 0)
    # "Only" is universal over list membership, not over possessed parts:
    # "contains only mammals" differs from "have only feathers".
    position = torch.arange(q.shape[1], device=q.device).unsqueeze(0)
    membership = w['group_membership'][q] > 0
    previous_member = membership.roll(1, dims=-1) * (position >= 1)
    previous_of = w['group_of'][q].roll(1, dims=-1) > 0
    earlier_member = membership.roll(2, dims=-1) * previous_of * (position >= 2)
    attached_only = (only * (previous_member + earlier_member)).sum(-1)
    membership_only = (only_count == 1) * (attached_only == 1)
    only_scope = (only_count == 0) + membership_only
    identity_requested = ((positive + negative) * w['group_identity']).sum(-1) > 0
    possession = w['group_possession'][q].sum(-1) > 0
    wrong_role = identity_requested * possession * (~membership_only)
    safe_question = question_pure * (w['group_forbidden'][q].sum(-1) == 0)
    safe_question = safe_question * only_scope * (~wrong_role) * (~unresolved) * (~contradictory) * (~union_scope)

    flat = o.reshape(-1, o.shape[-1])
    selected, named_coverage, occurrences = entity_name_spans(w, flat)
    entity_count = selected.sum(-1, keepdim=True)
    every_positive = (selected @ (w['trait_cards'] > 0).to(torch.float32)) == entity_count
    every_negative = (selected @ (w['trait_cards'] < 0).to(torch.float32)) == entity_count
    any_negative = selected @ (w['trait_cards'] < 0).to(torch.float32) > 0
    any_positive = selected @ (w['trait_cards'] > 0).to(torch.float32) > 0
    every_positive = every_positive.reshape(q.shape[0], 4, -1)
    every_negative = every_negative.reshape(q.shape[0], 4, -1)
    any_negative = any_negative.reshape(q.shape[0], 4, -1)
    any_positive = any_positive.reshape(q.shape[0], 4, -1)
    supported_count = (positive.unsqueeze(1) * every_positive + negative.unsqueeze(1) * every_negative).sum(-1)
    counterexamples = (positive.unsqueeze(1) * any_negative + negative.unsqueeze(1) * any_positive).sum(-1)
    entity_count = entity_count.reshape(q.shape[0], 4)
    pure_option = ((named_coverage + (w['group_option_filler'][flat] > 0) + (flat == 0)) > 0).amin(-1).reshape(q.shape[0], 4)
    distinct = entity_count == occurrences.reshape(q.shape[0], 4)
    cardinality = (entity_count >= 2) * (entity_count <= 4) * ((~pair) + (entity_count == 2))
    readable = pure_option * distinct * cardinality
    supported = (supported_count == count) * readable
    refuted = (counterexamples > 0) * readable
    # Closed, fully decided choices: unknown does not masquerade as false.
    active = request_noun * universal * safe_question * (count[:, 0] > 0)
    active = active * (readable.sum(-1) == 4) * (supported.sum(-1) == 1) * (refuted.sum(-1) == 3)
    evidence = supported.to(torch.float32) - refuted.to(torch.float32)
    return w['trait_gain'] * evidence * active.unsqueeze(-1)


def forward(w, q, o):
    n = q.shape[0]
    focused_q = question_focus(w, q)
    fb = bag(w, focused_q)
    whole = bag(w, q)
    qb = w['knobs'][7] * whole + (1 - w['knobs'][7]) * fb
    option_tokens = o.reshape(-1, o.shape[-1])
    ob = bag(w, option_tokens).reshape(n, 4, -1)
    denied, uncertain = denied_concepts(w, option_tokens)
    denied = denied.reshape(n, 4, -1)
    uncertain = uncertain.reshape(n, 4)
    # Unresolved double negation is not permission to affirm the raw bag.
    # All affirmative retrieval, role, and rank channels abstain on it too.
    affirmed = ob * (1 - denied) * (~uncertain).unsqueeze(-1)
    novel = 1 - w['knobs'][0] * affirmed.mean(1, keepdim=True)
    ow = affirmed * w['idf'] * novel * (1 - w['knobs'][1] * qb.unsqueeze(1))
    qw = qb * w['idf']
    qk = affinity(torch.sparse.mm(w['key'], qw.t()).t(), w['keymass'])
    qv = affinity(torch.sparse.mm(w['value'], qw.t()).t(), w['valuemass'])
    clashes = ow[:, :, w['clash_src']] * (affirmed[:, :, w['clash_dst']] == 0)
    ov = torch.sparse.mm(w['value'], ow.reshape(n * 4, -1).t()).t().reshape(n, 4, -1) - w['knobs'][10] * (clashes @ w['value_clash'])
    ok = torch.sparse.mm(w['key'], ow.reshape(n * 4, -1).t()).t().reshape(n, 4, -1) - w['knobs'][10] * (clashes @ w['key_clash'])
    option_scale = torch.sqrt(ow.sum(-1, keepdim=True) + 3)
    ov = ov.clamp_min(0) / torch.sqrt(w['valuemass'] + 2) / option_scale
    ok = ok.clamp_min(0) / torch.sqrt(w['keymass'] + 2) / option_scale
    directional = qk.unsqueeze(1).square() * ov.pow(w['knobs'][6])
    reverse = w['knobs'][2] * qv.unsqueeze(1).square() * ok.pow(w['knobs'][6])
    memory = torch.maximum(directional, reverse) * w['strength']
    strongest = memory.topk(3, dim=-1).values
    score = strongest[:, :, 0] + w['knobs'][3] * strongest[:, :, 1] + w['knobs'][4] * strongest[:, :, 2]
    score = score + logical_evidence(w, whole, ob, fb, denied, uncertain)
    score = score + assertion_evidence(w, whole, ob, focused_q, denied, uncertain)
    score = score + experimental_roles(w, q, affirmed, fb)
    score = score + slot_evidence(w, o, focused_q, whole, ob, denied, uncertain)
    score = score + food_web_evidence(w, q, o, focused_q, uncertain)
    score = score + proportional_evidence(w, q, o, focused_q, whole)
    score = score + trait_intersection(w, q, o, focused_q, fb, affirmed, uncertain)
    score = score + common_property_evidence(w, q, o, whole, ob)
    score = score + grouped_property_evidence(w, q, o, whole)
    negative = request_negation(w, focused_q)
    score = score * (1 - 2 * negative.unsqueeze(1))
    score = score + ordered_pathways(w, q, o, whole, fb, ob) * comparison_polarity(w, focused_q).unsqueeze(1)
    score = score + ordered_properties(w, whole, affirmed) * comparison_polarity(w, focused_q).unsqueeze(1)
    score = score + candidate_comparisons(w, q, o, focused_q, whole, ob) * comparison_polarity(w, focused_q).unsqueeze(1)
    score = score + candidate_numerical(w, q, o, focused_q, fb) * comparison_polarity(w, focused_q).unsqueeze(1)
    numeric = absolute_arithmetic(w, q, o, focused_q, whole, fb)
    numeric = numeric + collinear_forces(w, q, o, fb)
    numeric = numeric + direct_unit_conversion(w, q, o)
    # Exact-value evidence answers the same request as the verbal channels;
    # it must not favor the matching value in a direct 'not/except' request.
    score = score + numeric * (1 - 2 * negative.unsqueeze(1))
    return score
