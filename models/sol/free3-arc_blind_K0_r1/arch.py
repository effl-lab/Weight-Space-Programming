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
    return conversion, compatible


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
    prediction = lhs * rhs.clamp_min(0.000001).pow(w['math_op'])
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
    conversion, dimensional_match = option_measurement(w, o)
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
    units = physical_units(w, o) * (w['skip'][o] == 0).unsqueeze(-1)
    present = units.sum(-2) > 0
    unitless = present.sum(-1) == 0
    volume_compatible = (((present.sum(-1) == 1) * present[:, :, 1]) + unitless) * (w['per_word'][o].sum(-1) == 0)
    volume_scale = torch.where(unitless, torch.ones_like(numeric), units[:, :, :, 1].amax(-1))
    conversion, compatible = option_measurement(w, o)
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
    units = physical_units(w, o) * (w['skip'][o] == 0).unsqueeze(-1)
    present = units.sum(-2) > 0
    unitless = present.sum(-1) == 0
    volume_compatible = (((present.sum(-1) == 1) * present[:, :, 1]) + unitless) * (w['per_word'][o].sum(-1) == 0)
    volume_scale = torch.where(unitless, torch.ones_like(numeric), units[:, :, :, 1].amax(-1))
    conversion, compatible = option_measurement(w, o)
    observed_volume = numeric * volume_scale
    observed_density = numeric * conversion[:, :, 0]
    volume_error = (observed_volume - volume.unsqueeze(-1)).abs() / (volume.unsqueeze(-1) + 1)
    density_error = (observed_density - density.unsqueeze(-1)).abs() / (density.unsqueeze(-1) + 1)
    volume_score = torch.exp(-w['knobs'][9] * volume_error) * volume_compatible * wants_volume.unsqueeze(-1)
    density_score = torch.exp(-w['knobs'][9] * density_error) * compatible[:, :, 0] * (wants_density * mass_known).unsqueeze(-1)
    return w['knobs'][8] * (volume_score + density_score) * active.unsqueeze(-1) * (numeric_count == 1) * enough


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
    qk = affinity(qw @ w['key'].t(), w['keymass'])
    qv = affinity(qw @ w['value'].t(), w['valuemass'])
    clashes = ow[:, :, w['clash_src']] * (affirmed[:, :, w['clash_dst']] == 0)
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
    score = score + logical_evidence(w, whole, ob, fb, denied, uncertain)
    score = score + assertion_evidence(w, whole, ob, focused_q, denied, uncertain)
    score = score + experimental_roles(w, q, affirmed, fb)
    negative = request_negation(w, focused_q)
    score = score * (1 - 2 * negative.unsqueeze(1))
    score = score + ordered_properties(w, whole, affirmed) * comparison_polarity(w, focused_q).unsqueeze(1)
    numeric_safe = numerical_choice_mask(w, o)
    numeric = arithmetic(w, q, o, fb)
    numeric = numeric + nuclear_arithmetic(w, q, o, focused_q, whole)
    numeric = numeric + displacement_arithmetic(w, q, o, fb)
    numeric = numeric + geometric_arithmetic(w, q, o, fb)
    # Exact-value evidence answers the same request as the verbal channels;
    # it must not favor the matching value in a direct 'not/except' request.
    score = score + numeric * numeric_safe * (1 - 2 * negative.unsqueeze(1))
    return score
