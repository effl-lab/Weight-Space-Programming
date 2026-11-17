"""Fixed tensor inference: lexical concepts, relational retrieval and arithmetic.

Only torch/math imports and tensor computations depend on input contents.
"""
import torch
import math


def concepts(w, tokens):
    ids = w['token_concept'][tokens]
    features = torch.zeros((tokens.shape[0], w['feature_weight'].shape[0]), device=tokens.device)
    features.scatter_reduce_(1, ids, (ids > 0).to(torch.float32), reduce='amax', include_self=True)
    raw = w['token_raw_concept'][tokens]
    windows = torch.nn.functional.pad(raw, (0, 2)).unfold(1, 3, 1)
    matches = ((windows[:, :, None, :] == w['phrase_tokens'][None, None, :, :]) | (~w['phrase_mask'][None, None, :, :])).to(torch.float32).prod(dim=-1)
    phrase_presence = torch.amax(matches, dim=1)
    features.scatter_reduce_(1, w['phrase_id'][None, :].expand(tokens.shape[0], -1), phrase_presence, reduce='amax', include_self=True)
    inherited = features[:, w['tax_source']] * w['tax_strength'][None, :]
    features.scatter_reduce_(1, w['tax_target'][None, :].expand(tokens.shape[0], -1), inherited, reduce='amax', include_self=True)
    destinations = w['neighbors'].reshape(1, -1).expand(tokens.shape[0], -1)
    values = (features[:, :, None] * w['neighbor_weights'][None, :, :]).reshape(tokens.shape[0], -1)
    features.scatter_reduce_(1, destinations, values, reduce='amax', include_self=True)
    features[:, 0] = 0
    token_positions = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.float32)[None, :].expand(tokens.shape[0], -1)
    positions = torch.full_like(features, 1000.)
    positions.scatter_reduce_(1, ids, torch.where(ids > 0, token_positions, 1000.), reduce='amin', include_self=True)
    phrase_positions = torch.amin(torch.where(matches > 0, token_positions[:, :, None], 1000.), dim=1)
    positions.scatter_reduce_(1, w['phrase_id'][None, :].expand(tokens.shape[0], -1), phrase_positions, reduce='amin', include_self=True)
    neighbor_positions = positions[:, :, None].expand(-1, -1, 4).reshape(tokens.shape[0], -1)
    positions.scatter_reduce_(1, destinations, neighbor_positions, reduce='amin', include_self=True)
    positions[:, 0] = 1000.
    return features, positions


def unit_vectors(w, tokens):
    base = w['units'][tokens]
    cues = w['unit_cues'][tokens]
    following = torch.nn.functional.pad(base[:, 1:, :], (0, 0, 0, 1))
    following_cues = torch.nn.functional.pad(cues[:, 1:, :], (0, 0, 0, 1))
    cubic_prefix = cues[:, :, 0] * (following[:, :, 2] > 0).to(torch.float32)
    cubic_suffix = following_cues[:, :, 1] * (base[:, :, 2] > 0).to(torch.float32)
    volume = 1000000. * (cubic_prefix * following[:, :, 2].pow(3.) + cubic_suffix * base[:, :, 2].pow(3.))
    base = base * (1. - cubic_suffix[:, :, None]) + volume[:, :, None] * w['unit_axes'][1]
    next_unit = torch.nn.functional.pad(base[:, 1:, :], (0, 0, 0, 1))
    after_next_unit = torch.nn.functional.pad(base[:, 2:, :], (0, 0, 0, 2))
    per = following_cues[:, :, 2]
    denominator = next_unit * (1. - per[:, :, None]) + after_next_unit * per[:, :, None]
    squared_after_time = torch.nn.functional.pad(cues[:, 2:, 3], (0, 2)) * (1. - per) + torch.nn.functional.pad(cues[:, 3:, 3], (0, 3)) * per
    density = ((base[:, :, 0] > 0) & (denominator[:, :, 1] > 0)).to(torch.float32)
    velocity = ((base[:, :, 2] > 0) & (denominator[:, :, 3] > 0)).to(torch.float32)
    power = ((base[:, :, 5] > 0) & (denominator[:, :, 3] > 0)).to(torch.float32)
    acceleration = velocity * squared_after_time
    speed = velocity * (1. - squared_after_time)
    compound = density + velocity + power
    result = base * (1. - compound[:, :, None])
    result = result + (density * base[:, :, 0] / denominator[:, :, 1].clamp(min=1e-12))[:, :, None] * w['unit_axes'][9]
    result = result + (speed * base[:, :, 2] / denominator[:, :, 3].clamp(min=1e-12))[:, :, None] * w['unit_axes'][10]
    result = result + (power * base[:, :, 5] / denominator[:, :, 3].clamp(min=1e-12))[:, :, None] * w['unit_axes'][11]
    result = result + (acceleration * base[:, :, 2] / denominator[:, :, 3].clamp(min=1e-12).pow(2.))[:, :, None] * w['unit_axes'][12]
    return result


def number_values(w, tokens, units):
    values = w['numeric'][tokens]
    following = torch.nn.functional.pad(values[:, 1:], (0, 1), value=-1.)
    previous = torch.nn.functional.pad(values[:, :-1], (1, 0), value=-1.)
    fraction_divisor = torch.nn.functional.pad(w['decimal_divisor'][tokens[:, 1:]], (0, 1))
    unit_after_pair = torch.nn.functional.pad(torch.amax(units[:, 2:, :], dim=2), (0, 2))
    padding_after_pair = torch.nn.functional.pad((tokens[:, 2:] == 0).to(torch.float32), (0, 2), value=1.)
    decimal_pair = ((values >= 0) & (following >= 0) & (previous < 0) & (fraction_divisor > 0) & ((fraction_divisor <= 100) | (values == 0)) & ((unit_after_pair > 0) | (padding_after_pair > 0))).to(torch.float32)
    joined = values + decimal_pair * following / fraction_divisor.clamp(min=1.)
    consumed_fraction = torch.nn.functional.pad(decimal_pair[:, :-1], (1, 0))
    result = torch.where(consumed_fraction > 0, -1., joined)
    shifted_units = torch.nn.functional.pad(units[:, 1:, :], (0, 0, 0, 1))
    next_units = torch.nn.functional.pad(units[:, 2:, :], (0, 0, 0, 2))
    adjacent_units = shifted_units * (1. - decimal_pair[:, :, None]) + next_units * decimal_pair[:, :, None]
    return result, adjacent_units


def measurements(w, tokens):
    units = unit_vectors(w, tokens)
    numbers, attached_units = number_values(w, tokens, units)
    valid = (numbers >= 0).to(torch.float32)
    counts = ((attached_units > 0).to(torch.float32) * valid[:, :, None]).sum(dim=1)
    quantities = (numbers.clamp(min=0)[:, :, None] * attached_units * valid[:, :, None]).sum(dim=1) / counts.clamp(min=1.)
    return quantities, counts, numbers, attached_units


def calculate(w, quantities, counts):
    required = (w['formula_exponents'] != 0).to(torch.float32)
    missing = (counts != 1).to(torch.float32) @ required
    division_by_zero = (quantities <= 0).to(torch.float32) @ (w['formula_exponents'] < 0).to(torch.float32)
    given_output = counts[:, w['formula_output']] > 0
    calculated = torch.exp(torch.log(quantities.clamp(min=1e-12)) @ w['formula_exponents']) * w['formula_coefficient'][None, :]
    zero_numerator = (quantities == 0).to(torch.float32) @ (w['formula_exponents'] > 0).to(torch.float32)
    calculated = calculated * (zero_numerator == 0).to(torch.float32)
    gates = ((missing == 0) & (division_by_zero == 0) & (~given_output) & torch.isfinite(calculated)).to(torch.float32)
    return torch.nan_to_num(calculated, nan=0., posinf=0., neginf=0.), gates


def arithmetic(w, q, o, qfeatures):
    quantities, counts, numbers, attached_units = measurements(w, q)
    calculated, availability = calculate(w, quantities, counts)
    target = torch.amax(qfeatures[:, w['formula_topics']], dim=2)
    gates = availability * target

    flat_options = o.reshape(-1, o.shape[-1])
    ounits = unit_vectors(w, flat_options)
    onumbers, option_attached = number_values(w, flat_options, ounits)
    numeric_count = (onumbers >= 0).to(torch.float32).sum(dim=1)
    value = onumbers.clamp(min=0).sum(dim=1) / numeric_count.clamp(min=1.)
    unit_count = ((option_attached > 0).to(torch.float32) * (onumbers >= 0).to(torch.float32)[:, :, None]).sum(dim=1)
    unit_scale = (option_attached * (onumbers >= 0).to(torch.float32)[:, :, None]).sum(dim=1) / unit_count.clamp(min=1.)
    unit_scale = torch.where(unit_count > 0, unit_scale, 1.)
    candidate = (value[:, None] * unit_scale[:, w['formula_output']]).reshape(q.shape[0], 4, -1)
    dimension_matches = ((unit_count[:, w['formula_output']] > 0) | (unit_count.sum(dim=1, keepdim=True) == 0)).to(torch.float32)
    option_valid = ((numeric_count == 1).to(torch.float32)[:, None] * dimension_matches).reshape(q.shape[0], 4, -1)
    error = torch.abs(torch.log((candidate + 1e-6) / (calculated[:, None, :] + 1e-6)))
    agreement = torch.exp(-5. * error)
    return 3.0 * torch.amax(agreement * option_valid * gates[:, None, :], dim=2)


def comparisons(w, q, o, qfeatures, option_positions, recent):
    batch = q.shape[0]
    literal = (option_positions.reshape(-1, option_positions.shape[-1]) < 1000.).to(torch.float32)
    groups = torch.zeros((literal.shape[0], w['property_group_map'].shape[0]), device=q.device)
    groups.scatter_reduce_(1, w['property_entry_groups'][None, :].expand(literal.shape[0], -1), literal[:, w['property_entry_ids']], reduce='amax', include_self=True)
    named_count = groups @ w['property_group_map']
    named_values = (groups @ w['property_group_values']) / named_count.clamp(min=1.)
    named_valid = named_count == 1

    quantities, counts, numbers, attached_units = measurements(w, o.reshape(-1, o.shape[-1]))
    calculated, availability = calculate(w, quantities, counts)
    derived_count = availability @ w['formula_dimensions']
    derived_value = ((calculated * availability) @ w['formula_dimensions']) / derived_count.clamp(min=1.)
    numerical_value = torch.where(counts == 1, quantities, derived_value)
    numerical_valid = (counts == 1) | ((counts == 0) & (derived_count == 1))
    numeric_for_property = numerical_valid[:, w['property_dimension']] & (w['property_numeric'][None, :] > 0)
    # Four bare numeric choices share the question's implied unit, so their
    # relative order is meaningful. Never mix that assumption with explicit
    # units or with numbers embedded in descriptions of named objects.
    bare_tokens = (((o > 0) & (w['numeric'][o] < 0)).sum(dim=2) == 0)
    single_number = ((numbers >= 0).sum(dim=1) == 1).reshape(batch, 4)
    all_bare = (bare_tokens & single_number).to(torch.float32).prod(dim=1)
    bare_for_property = (all_bare[:, None, None].expand(-1, 4, w['property_numeric'].shape[0]).reshape(-1, w['property_numeric'].shape[0]) > 0) & (w['property_numeric'][None, :] > 0)
    numerical_property_values = torch.where(bare_for_property, numbers.clamp(min=0).sum(dim=1, keepdim=True), numerical_value[:, w['property_dimension']])
    numeric_for_property = numeric_for_property | bare_for_property
    ambiguous_numeric = (counts[:, w['property_dimension']] > 1) & (w['property_numeric'][None, :] > 0)
    values = torch.where(numeric_for_property, numerical_property_values, named_values).reshape(batch, 4, -1)
    valid = (numeric_for_property | (named_valid & (~ambiguous_numeric))).reshape(batch, 4, -1)
    negated_option = torch.amax(w['oneg'][o], dim=2)
    complete = (valid & (negated_option[:, :, None] == 0)).to(torch.float32).prod(dim=1)

    # The nearest interrogative's content controls attribute selection; earlier
    # context is retained weakly for subject names such as planets.
    local_features, local_positions = concepts(w, q * recent.to(q.dtype))
    context_features = torch.maximum(local_features, .60 * qfeatures)
    coverage = (context_features @ w['comparison_query']).clamp(0., 1.)
    probability = w['probability_words'][q]
    next_likely = torch.nn.functional.pad(probability[:, 1:, 1], (0, 1))
    probability_adverb = probability[:, :, 0] * next_likely
    cue_scope = recent * (1. - probability_adverb)
    positive = torch.amax(w['comparison_positive'][q] * cue_scope[:, :, None], dim=1)
    negative = torch.amax(w['comparison_negative'][q] * cue_scope[:, :, None], dim=1)
    implicit = ((positive + negative) == 0).to(torch.float32) * w['comparison_default'][None, :] * (coverage >= .95).to(torch.float32)
    specificity = 1. + .25 * torch.log1p(w['comparison_size'])
    rule = coverage.pow(6.) * (positive - negative + implicit) * specificity[None, :]
    property_positive = torch.zeros((batch, w['property_numeric'].shape[0]), device=q.device)
    property_negative = torch.zeros_like(property_positive)
    destinations = w['comparison_property'][None, :].expand(batch, -1)
    property_positive.scatter_reduce_(1, destinations, rule.clamp(min=0), reduce='amax', include_self=True)
    property_negative.scatter_reduce_(1, destinations, (-rule).clamp(min=0), reduce='amax', include_self=True)
    logical_not = torch.amax(w['comparison_not'][q] * recent, dim=1)
    least_likely = torch.amax(probability[:, :, 2] * next_likely * recent, dim=1)
    strength = (property_positive - property_negative) * complete * ((1. - 2. * logical_not) * (1. - 2. * least_likely))[:, None]

    # Rank only comparable, known values; preserve equal-valued choices.
    difference = values[:, :, None, :] - values[:, None, :, :]
    tolerance = 1e-5 * torch.maximum(torch.abs(values[:, :, None, :]), torch.abs(values[:, None, :, :])).clamp(min=1e-8)
    ordering = (difference > tolerance).to(torch.float32) - (difference < -tolerance).to(torch.float32)
    rank = ordering.sum(dim=2) / 3.
    strongest = torch.amax(torch.abs(strength), dim=1, keepdim=True)
    selected = ((torch.abs(strength) >= strongest - 1e-6) & (torch.abs(strength) > 1e-6)).to(torch.float32)
    return 2.0 * (rank * (strength * selected)[:, None, :]).sum(dim=2) / selected.sum(dim=1, keepdim=True).clamp(min=1.)


def forward(w, q, o):
    qf, qp = concepts(w, q)
    option_features, option_positions = concepts(w, o.reshape(-1, o.shape[-1]))
    of = option_features.reshape(q.shape[0], 4, -1)
    op = option_positions.reshape(q.shape[0], 4, -1)

    # Context coverage has a steep gate: several mutually consistent clues
    # outrank a solitary word in a broad topic.
    coverage = (qf @ w['qmat']).clamp(0., 1.)
    activation = coverage.pow(6.) * (0.70 + 0.30 * torch.sqrt(w['qmass']))[None, :]
    # Prefer well-covered contexts without turning weak evidence into certainty.
    retrieval_priority = activation * w['context_confidence'][None, :]
    strongest = torch.amax(retrieval_priority, dim=1, keepdim=True)
    activation = activation * (0.40 + 0.60 * (retrieval_priority / strongest.clamp(min=1e-6)).pow(2.))
    context = activation[:, w['acontext']] * w['confidence'][None, :]

    # Candidate words shared by every option or already supplied by the
    # question provide less discriminating evidence than novel answer content.
    distinct = (1.0 - 0.70 * of.mean(dim=1)) * (1.0 - 0.45 * qf)
    raw_coverage = (of @ w['amat']).clamp(0., 1.)
    distinctive_coverage = ((of * distinct[:, None, :]) @ w['amat']).clamp(0., 1.)
    evidence = raw_coverage.pow(2.5) * (0.30 + 0.70 * distinctive_coverage / raw_coverage.clamp(min=1e-6))
    evidence = evidence * (0.85 + 0.15 * torch.sqrt(w['amass']))[None, None, :]
    evidence = evidence * context[:, None, :]

    # An assertion about the absence of a property has the opposite polarity
    # to one about its presence. The polarity is also reversed by a negated
    # option, independently of whether the question asks for an exception.
    option_negative = torch.amax(w['oneg'][o], dim=2)
    polarity = w['asign'][None, None, :] * w['apolarity'][None, None, :] * (1. - 2. * option_negative[:, :, None])
    positions = torch.arange(q.shape[1], device=q.device)[None, :]
    length = (q > 0).sum(dim=1, keepdim=True)
    last_question_start = torch.amax(torch.where(w['question_start'][q] > 0, positions, -1), dim=1, keepdim=True)
    recent = torch.where(last_question_start >= 0, positions >= last_question_start, positions >= length - 40).to(torch.float32)
    question_negative = torch.maximum(torch.amax(w['qneg'][q] * recent, dim=1), torch.amax(w['qexcept'][q] * recent, dim=1))
    direct = w['context_direct'][w['acontext']]
    question_sign = 1. - 2. * question_negative[:, None] * (1. - direct[None, :])
    polarity = polarity * question_sign[:, None, :]

    positive = evidence * (polarity > 0).to(torch.float32)
    negative = evidence * (polarity < 0).to(torch.float32)
    support = (torch.topk(positive, 4, dim=2).values * w['top_weights']).sum(dim=2)
    contradiction = (torch.topk(negative, 4, dim=2).values * w['top_weights']).sum(dim=2)
    logits = support - 0.85 * contradiction
    logits = logits + arithmetic(w, q, o, qf)

    # Directed transformations: lexical bags alone cannot distinguish
    # electrical -> mechanical from mechanical -> electrical.
    source_pos = op[:, :, w['rsource']]
    target_pos = op[:, :, w['rtarget']]
    left = torch.minimum(source_pos, target_pos)
    right = torch.maximum(source_pos, target_pos)
    opositions = torch.arange(o.shape[2], device=o.device)[None, None, :, None]
    from_between = torch.amax(w['from_token'][o][:, :, :, None] * ((opositions > left[:, :, None, :]) & (opositions < right[:, :, None, :])).to(torch.float32), dim=2)
    direction = torch.sign(target_pos - source_pos) * (1. - 2. * from_between)
    relationship_present = ((source_pos < 1000.) & (target_pos < 1000.)).to(torch.float32)
    relationship_context = (qf @ w['rquery']).clamp(0., 1.).pow(6.)
    relationship = direction * relationship_present * relationship_context[:, None, :]
    relation_support = torch.amax(relationship.clamp(min=0), dim=2)
    relation_conflict = torch.amax((-relationship).clamp(min=0), dim=2)
    logits = logits + 1.8 * (relation_support - relation_conflict) * (1. - 2. * question_negative[:, None]) * (1. - 2. * option_negative)
    qsource = qp[:, w['rsource']]
    qtarget = qp[:, w['rtarget']]
    qleft = torch.minimum(qsource, qtarget)
    qright = torch.maximum(qsource, qtarget)
    qfrom_between = torch.amax(w['from_token'][q][:, :, None] * ((positions[:, :, None] > qleft[:, None, :]) & (positions[:, :, None] < qright[:, None, :])).to(torch.float32), dim=1)
    qdirection = torch.sign(qtarget - qsource) * (1. - 2. * qfrom_between)
    qrelation_present = ((qsource < 1000.) & (qtarget < 1000.)).to(torch.float32)
    reverse_evidence = (of @ w['rquery']).clamp(0., 1.).pow(4.) * (qdirection * qrelation_present)[:, None, :]
    reverse_score = torch.amax(reverse_evidence.clamp(min=0), dim=2) - torch.amax((-reverse_evidence).clamp(min=0), dim=2)
    logits = logits + 1.8 * reverse_score * (1. - 2. * question_negative[:, None]) * (1. - 2. * option_negative)

    # Equality and opposition of magnetic poles / electric charges require
    # identity and multiplicity, not merely bag-of-words association.
    iq = qf[:, w['interaction_ids']]
    io = of[:, :, w['interaction_ids']]
    counts = (w['token_raw_concept'][q][:, :, None] == w['interaction_ids'][None, None, :]).to(torch.float32).sum(dim=1)
    pair = ((iq[:, 6] >= .99) | (iq[:, 7] >= .99)).to(torch.float32)
    magnetic_pair = torch.maximum(pair, (torch.amax(counts[:, 2:4], dim=1) >= 2).to(torch.float32))
    electric_pair = torch.maximum(pair, (torch.amax(counts[:, 4:6], dim=1) >= 2).to(torch.float32))
    north = (iq[:, 2] > .9).to(torch.float32)
    south = (iq[:, 3] > .9).to(torch.float32)
    plus = (iq[:, 4] > .9).to(torch.float32)
    minus = (iq[:, 5] > .9).to(torch.float32)
    same_m = torch.maximum(torch.abs(north - south) * magnetic_pair, iq[:, 8])
    opposite_m = torch.maximum(north * south, iq[:, 9])
    same_e = torch.maximum(torch.abs(plus - minus) * electric_pair, iq[:, 8])
    opposite_e = torch.maximum(plus * minus, iq[:, 9])
    force_direction = iq[:, 0] * (same_m - opposite_m) + iq[:, 1] * (same_e - opposite_e)
    interaction = force_direction[:, None] * (io[:, :, 11] - io[:, :, 10])
    magnetic_target = iq[:, 0] * (north - south) * (iq[:, 11] - iq[:, 10])
    electric_target = iq[:, 1] * (plus - minus) * (iq[:, 11] - iq[:, 10])
    interaction = interaction + magnetic_target[:, None] * (io[:, :, 2] - io[:, :, 3]) + electric_target[:, None] * (io[:, :, 4] - io[:, :, 5])
    logits = logits + 1.8 * interaction * (1. - 2. * question_negative[:, None]) * (1. - 2. * option_negative)

    # The first explicitly named class in the final question is a modest
    # category constraint. Inferred classes are never treated as question text.
    class_positions = qp[:, w['tax_classes']]
    class_prompt = torch.amax(w['class_question'][q] * (positions == last_question_start).to(torch.float32), dim=1, keepdim=True)
    eligible = (class_positions >= last_question_start) & (class_positions <= last_question_start + 10) & (class_positions < 1000.) & (class_prompt > 0)
    first_class = torch.amin(torch.where(eligible, class_positions, 1000.), dim=1, keepdim=True)
    requested = ((class_positions == first_class) & eligible).to(torch.float32)
    membership = (of[:, :, w['tax_classes']] * requested[:, None, :]).sum(dim=2) / requested.sum(dim=1, keepdim=True).clamp(min=1.)
    logits = logits + .75 * membership * (1. - 2. * question_negative[:, None]) * (1. - 2. * option_negative)
    logits = logits + comparisons(w, q, o, qf, op, recent)
    return logits
