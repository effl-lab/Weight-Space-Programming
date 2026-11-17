"""Fixed tensor inference: lexical concepts, relational retrieval and arithmetic.

Only torch/math imports and tensor computations depend on input contents.
"""
import torch
import math


def project(x, matrix):
    # Sparse matrix multiplication over any fixed leading batch dimensions.
    result = torch.sparse.mm(matrix.transpose(0, 1), x.reshape(-1, x.shape[-1]).transpose(0, 1)).transpose(0, 1)
    return result.reshape(x.shape[:-1] + (matrix.shape[1],))


def concepts(w, tokens):
    ids = w['token_concept'][tokens]
    features = torch.zeros((tokens.shape[0], w['feature_weight'].shape[0]), device=tokens.device)
    features.scatter_reduce_(1, ids, (ids > 0).to(torch.float32), reduce='amax', include_self=True)
    raw = w['token_raw_concept'][tokens]
    windows = torch.nn.functional.pad(raw, (0, 2)).unfold(1, 3, 1)
    token_positions = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.float32)[None, :].expand(tokens.shape[0], -1)
    positions = torch.full_like(features, 1000.)
    entity_features = torch.zeros_like(features)
    consumed = torch.zeros_like(token_positions)
    chunk = (w['phrase_id'].shape[0] + 7) // 8
    for part in range(8):
        begin = part * chunk
        end = (part + 1) * chunk
        matches = ((windows[:, :, None, :] == w['phrase_tokens'][None, None, begin:end, :]) | (~w['phrase_mask'][None, None, begin:end, :])).to(torch.float32).prod(dim=-1)
        phrase_presence = torch.amax(matches, dim=1)
        destinations_phrase = w['phrase_id'][None, begin:end].expand(tokens.shape[0], -1)
        features.scatter_reduce_(1, destinations_phrase, phrase_presence, reduce='amax', include_self=True)
        entity_features.scatter_reduce_(1, destinations_phrase, phrase_presence, reduce='amax', include_self=True)
        phrase_positions = torch.amin(torch.where(matches > 0, token_positions[:, :, None], 1000.), dim=1)
        positions.scatter_reduce_(1, destinations_phrase, phrase_positions, reduce='amin', include_self=True)
        for offset in range(3):
            occupied = torch.amax(matches * w['phrase_entity'][None, None, begin:end, offset], dim=2)
            consumed = torch.maximum(consumed, torch.nn.functional.pad(occupied, (offset, 0))[:, :tokens.shape[1]])
    standalone = (ids > 0).to(torch.float32) * (1. - consumed)
    entity_features.scatter_reduce_(1, ids, standalone, reduce='amax', include_self=True)
    positions.scatter_reduce_(1, ids, torch.where(standalone > 0, token_positions, 1000.), reduce='amin', include_self=True)
    # A class word consumed by an entity name is not a literal membership:
    # 'sea star' is not a star, though non-class component terms can still aid
    # associative retrieval. True inherited memberships are added afterwards.
    features = features * (1. - w['class_mask'][None, :] * (1. - entity_features))
    inherited = entity_features[:, w['tax_source']] * w['tax_strength'][None, :]
    features.scatter_reduce_(1, w['tax_target'][None, :].expand(tokens.shape[0], -1), inherited, reduce='amax', include_self=True)
    destinations = w['neighbors'].reshape(1, -1).expand(tokens.shape[0], -1)
    values = (features[:, :, None] * w['neighbor_weights'][None, :, :]).reshape(tokens.shape[0], -1)
    features.scatter_reduce_(1, destinations, values, reduce='amax', include_self=True)
    features[:, 0] = 0
    neighbor_positions = positions[:, :, None].expand(-1, -1, 4).reshape(tokens.shape[0], -1)
    positions.scatter_reduce_(1, destinations, neighbor_positions, reduce='amin', include_self=True)
    positions[:, 0] = 1000.
    return features, positions


def negation_mask(w, tokens):
    following = torch.nn.functional.pad(w['nonnegating_follow'][tokens[..., 1:]], (0, 1))
    return 1. - w['literal_not'][tokens] * following


def option_negation(w, tokens):
    return torch.amax(w['oneg'][tokens] * negation_mask(w, tokens), dim=-1)


def answer_scope(w, q, recent):
    # A why-clause supplies the premise to explain, not the polarity of the
    # requested answer. Keep a preceding main-clause negation in expressions
    # such as 'which statement does not explain why ...'. This deliberately
    # recognizes only standalone why and adjacent explain/reason-why bridges.
    positions = torch.arange(q.shape[1], device=q.device)[None, :]
    first = torch.amin(torch.where(recent > 0, positions, 1000), dim=1, keepdim=True)
    previous = torch.nn.functional.pad(w['token_raw_concept'][q[:, :-1]], (1, 0))
    bridge = torch.amax((previous[:, :, None] == w['reason_bridge'][None, None, :]).to(torch.float32), dim=2)
    premise = (q == w['why_token']) & ((positions == first) | (bridge > 0)) & (recent > 0)
    boundary = torch.amin(torch.where(premise, positions, 1000), dim=1, keepdim=True)
    return recent * (positions < boundary).to(torch.float32)


def question_scope(w, q):
    positions = torch.arange(q.shape[1], device=q.device)[None, :]
    length = (q > 0).sum(dim=1, keepdim=True)
    main = torch.amax(torch.where(w['question_main'][q] > 0, positions, -1), dim=1, keepdim=True)
    explanatory = torch.amax(torch.where(w['question_explain'][q] > 0, positions, -1), dim=1, keepdim=True)
    other = torch.amax(torch.where(w['question_start'][q] > 0, positions, -1), dim=1, keepdim=True)
    # A subordinate why/how/when must not erase 'not' in a main which/what
    # clause. Prefer the explicit answer-requesting interrogative category.
    start = torch.where(main >= 0, main, torch.where(explanatory >= 0, explanatory, other))
    recent = torch.where(start >= 0, positions >= start, positions >= length - 40).to(torch.float32)
    request_scope = answer_scope(w, q, recent)
    ordinary = torch.amax(w['qneg'][q] * negation_mask(w, q) * request_scope, dim=1)
    exception = torch.amax(w['qexcept'][q] * request_scope, dim=1)
    # Avoidance is not inherently question negation: an adaptation can help an
    # animal avoid predators. Only a nearby directive such as 'should avoid'
    # activates the prohibited-action interpretation.
    separation = positions[:, :, None] - positions[:, None, :]
    eligible_modal = (separation > 0) & (separation <= 6) & (w['directive_modal'][q][:, None, :] > 0) & (request_scope[:, None, :] > 0)
    nearest_modal = torch.amax(torch.where(eligible_modal, positions[:, None, :], -1), dim=2)
    previous_to = torch.nn.functional.pad(w['purpose_to'][q[:, :-1]], (1, 0))
    directive_avoid = torch.amax(w['avoid_token'][q] * (nearest_modal >= 0).to(torch.float32) * (1. - previous_to) * request_scope, dim=1)
    negative = torch.maximum(torch.maximum(ordinary, exception), directive_avoid)
    return positions, start, recent, negative


def parsed_units(w, tokens):
    base = w['units'][tokens]
    cues = w['unit_cues'][tokens]
    following = torch.nn.functional.pad(base[:, 1:, :], (0, 0, 0, 1))
    following_cues = torch.nn.functional.pad(cues[:, 1:, :], (0, 0, 0, 1))
    cubic_prefix = cues[:, :, 0] * (following[:, :, 2] > 0).to(torch.float32)
    cubic_suffix = following_cues[:, :, 1] * (base[:, :, 2] > 0).to(torch.float32)
    volume = 1000000. * (cubic_prefix * following[:, :, 2].pow(3.) + cubic_suffix * base[:, :, 2].pow(3.))
    unsupported_cube = following_cues[:, :, 1] * (1. - cubic_suffix)
    base = base * (1. - cubic_suffix[:, :, None]) * (1. - unsupported_cube[:, :, None]) + volume[:, :, None] * w['unit_axes'][1]
    consumed_length = torch.nn.functional.pad(cubic_prefix[:, :-1], (1, 0))
    base = base * (1. - consumed_length[:, :, None])
    base_length = torch.where(cubic_prefix + cubic_suffix > 0, 2., (torch.amax(base, dim=2) > 0).to(torch.float32))
    next_unit = torch.nn.functional.pad(base[:, 1:, :], (0, 0, 0, 1))
    after_next_unit = torch.nn.functional.pad(base[:, 2:, :], (0, 0, 0, 2))
    per = following_cues[:, :, 2]
    denominator = next_unit * (1. - per[:, :, None]) + after_next_unit * per[:, :, None]
    squared_after_time = torch.nn.functional.pad(cues[:, 2:, 3], (0, 2)) * (1. - per) + torch.nn.functional.pad(cues[:, 3:, 3], (0, 3)) * per
    density = ((base[:, :, 0] > 0) & (denominator[:, :, 1] > 0)).to(torch.float32) * (1. - squared_after_time)
    velocity = ((base[:, :, 2] > 0) & (denominator[:, :, 3] > 0)).to(torch.float32)
    power = ((base[:, :, 5] > 0) & (denominator[:, :, 3] > 0)).to(torch.float32) * (1. - squared_after_time)
    acceleration = velocity * squared_after_time
    speed = velocity * (1. - squared_after_time)
    compound = density + velocity + power
    # Unsupported powers/quotients are not the unmodified numerator unit.
    # Keep raw time units above for acceleration's squared denominator, but
    # reject standalone seconds squared and all other unsupported squares.
    denominator_absent = (torch.amax(denominator, dim=2) == 0).to(torch.float32)
    result = base * (1. - compound[:, :, None]) * (1. - per[:, :, None]) * denominator_absent[:, :, None] * (1. - following_cues[:, :, 3:4])
    result = result + (density * base[:, :, 0] / denominator[:, :, 1].clamp(min=1e-12))[:, :, None] * w['unit_axes'][9]
    result = result + (speed * base[:, :, 2] / denominator[:, :, 3].clamp(min=1e-12))[:, :, None] * w['unit_axes'][10]
    result = result + (power * base[:, :, 5] / denominator[:, :, 3].clamp(min=1e-12))[:, :, None] * w['unit_axes'][11]
    result = result + (acceleration * base[:, :, 2] / denominator[:, :, 3].clamp(min=1e-12).pow(2.))[:, :, None] * w['unit_axes'][12]
    denominator_length = torch.nn.functional.pad(base_length[:, 1:], (0, 1)) * (1. - per) + torch.nn.functional.pad(base_length[:, 2:], (0, 2)) * per
    span = (1. - compound) * base_length + compound * (1. + per + denominator_length + acceleration)
    span = span * (torch.amax(result, dim=2) > 0).to(torch.float32)
    # A supported prefix is not the whole unit when a further quotient,
    # exponent, or unit follows. In particular m^3/s is not plain volume.
    positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
    end = (positions + span.to(torch.int64)).clamp(max=tokens.shape[1])
    syntax = torch.maximum((torch.amax(w['units'][tokens], dim=2) > 0).to(torch.float32), torch.amax(cues, dim=2))
    continuation = torch.nn.functional.pad(syntax, (0, 1)).gather(1, end)
    result = result * (1. - continuation[:, :, None])
    span = span * (torch.amax(result, dim=2) > 0).to(torch.float32)
    return result, span


def unit_vectors(w, tokens):
    result, span = parsed_units(w, tokens)
    return result


def number_values(w, tokens, units):
    values = w['numeric'][tokens]
    following = torch.nn.functional.pad(values[:, 1:], (0, 1), value=-1.)
    previous = torch.nn.functional.pad(values[:, :-1], (1, 0), value=-1.)
    fraction_divisor = torch.nn.functional.pad(w['decimal_divisor'][tokens[:, 1:]], (0, 1))
    unit_after_pair = torch.nn.functional.pad(torch.amax(units[:, 2:, :], dim=2), (0, 2))
    padding_after_pair = torch.nn.functional.pad((tokens[:, 2:] == 0).to(torch.float32), (0, 2), value=1.)
    decimal_pair = ((w['numeric_digit'][tokens] > 0) & (following >= 0) & (previous < 0) & (fraction_divisor > 0) & ((fraction_divisor <= 100) | (values == 0)) & ((unit_after_pair > 0) | (padding_after_pair > 0))).to(torch.float32)
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


def formula_applicability(w, tokens, request):
    # Necessary physical assumptions are distinct from dimensional validity.
    # Explicit conflicting cues withhold a formula, not invert its result.
    scope = 1. - w['formula_condition_request'][None, None, :] + request[:, :, None] * w['formula_condition_request'][None, None, :]
    conflicts = torch.amax(w['formula_condition_tokens'][tokens] * scope, dim=1)
    return 1. - (conflicts @ w['formula_condition_map']).clamp(0., 1.)


def arithmetic(w, q, o, qfeatures):
    quantities, counts, numbers, attached_units = measurements(w, q)
    calculated, availability = calculate(w, quantities, counts)
    _, _, recent, negative_question = question_scope(w, q)
    request = answer_scope(w, q, recent)
    literal_topic = (w['token_raw_concept'][q][:, :, None, None] == w['formula_topics'][None, None, :, :]) & (w['formula_topics'][None, None, :, :] > 0)
    target = torch.amax(literal_topic.to(torch.float32) * request[:, :, None, None], dim=(1, 3))
    gates = availability * target * formula_applicability(w, q, request)

    flat_options = o.reshape(-1, o.shape[-1])
    ounits, unit_spans = parsed_units(w, flat_options)
    onumbers, option_attached = number_values(w, flat_options, ounits)
    numeric_count = (onumbers >= 0).to(torch.float32).sum(dim=1)
    value = onumbers.clamp(min=0).sum(dim=1) / numeric_count.clamp(min=1.)
    unit_count = ((option_attached > 0).to(torch.float32) * (onumbers >= 0).to(torch.float32)[:, :, None]).sum(dim=1)
    unit_scale = (option_attached * (onumbers >= 0).to(torch.float32)[:, :, None]).sum(dim=1) / unit_count.clamp(min=1.)
    unit_scale = torch.where(unit_count > 0, unit_scale, 1.)
    candidate = (value[:, None] * unit_scale[:, w['formula_output']]).reshape(q.shape[0], 4, -1)
    # Only genuinely bare numeric choices may borrow an implied unit. An
    # unrecognized unit such as kelvin must not become dimensionless evidence.
    positions = torch.arange(flat_options.shape[1], device=o.device)[None, :]
    numeric_token = (w['numeric'][flat_options] >= 0)
    first_number = torch.amin(torch.where(numeric_token, positions, 1000), dim=1, keepdim=True)
    wrapper = (w['numeric_wrapper'][flat_options] > 0) & (positions < first_number)
    bare_numeric = ((numeric_token | wrapper | (flat_options == 0)).to(torch.float32).prod(dim=1) > 0)
    # Account for the complete unit span attached to the sole numeric value.
    # Two adjacent digit tokens may encode a decimal after tokenization.
    paired = torch.nn.functional.pad(numeric_token[:, 1:], (0, 1)) & (onumbers >= 0)
    unit_start = positions + 1 + paired.to(torch.int64)
    lengths = torch.nn.functional.pad(unit_spans, (0, 2)).gather(1, unit_start)
    attached = (torch.amax(option_attached, dim=2) > 0) & (onumbers >= 0)
    covered = (positions[:, None, :] >= unit_start[:, :, None]) & (positions[:, None, :] < (unit_start + lengths)[:, :, None]) & attached[:, :, None]
    covered = torch.amax(covered.to(torch.float32), dim=1) > 0
    complete_option = (numeric_token | wrapper | covered | (flat_options == 0)).to(torch.float32).prod(dim=1)
    dimension_matches = ((unit_count[:, w['formula_output']] > 0) | bare_numeric[:, None]).to(torch.float32)
    affirmative_option = 1. - option_negation(w, flat_options)
    option_valid = (((numeric_count == 1).to(torch.float32) * affirmative_option * complete_option)[:, None] * dimension_matches).reshape(q.shape[0], 4, -1)
    error = torch.abs(torch.log((candidate + 1e-6) / (calculated[:, None, :] + 1e-6)))
    agreement = torch.exp(-5. * error)
    return 3.0 * torch.amax(agreement * option_valid * gates[:, None, :], dim=2) * (1. - negative_question[:, None])


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
    described = o.reshape(-1, o.shape[-1])
    availability = availability * formula_applicability(w, described, torch.ones_like(described, dtype=torch.float32))
    # The requested physical kind also constrains derived comparison values.
    # Comparative operators (highest/lowest) rank the options; here they do
    # not ask for an unobserved maximum/minimum during a single journey.
    comparative = torch.maximum(torch.amax(w['comparison_positive'][q], dim=2), torch.amax(w['comparison_negative'][q], dim=2))
    physical_request = answer_scope(w, q, recent) * (1. - comparative)
    question_applicability = formula_applicability(w, q, physical_request)
    availability = availability * question_applicability[:, None, :].expand(-1, 4, -1).reshape(-1, availability.shape[1])
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
    negated_option = option_negation(w, o)
    complete = (valid & (negated_option[:, :, None] == 0)).to(torch.float32).prod(dim=1)

    # The nearest interrogative's content controls attribute selection; earlier
    # context is retained weakly for subject names such as planets.
    request_scope = answer_scope(w, q, recent)
    local_features, local_positions = concepts(w, q * request_scope.to(q.dtype))
    context_features = torch.maximum(local_features, .60 * qfeatures)
    coverage = (context_features @ w['comparison_query']).clamp(0., 1.)
    # All selector concepts must be recognized somewhere in the question.
    # A generic 'element' alone cannot select the 'element proton' rule and
    # turn an unsupported isotope mass-number question into an atomic-number
    # ranking. Scoped coverage still controls strength after identification.
    identified = (((qfeatures > 0).to(torch.float32) @ w['comparison_query']) >= .999).to(torch.float32)
    probability = w['probability_words'][q]
    next_likely = torch.nn.functional.pad(probability[:, 1:, 1], (0, 1))
    probability_adverb = probability[:, :, 0] * next_likely
    cue_scope = request_scope * (1. - probability_adverb)
    positive = torch.amax(w['comparison_positive'][q] * cue_scope[:, :, None], dim=1)
    negative = torch.amax(w['comparison_negative'][q] * cue_scope[:, :, None], dim=1)
    implicit = ((positive + negative) == 0).to(torch.float32) * w['comparison_default'][None, :] * (coverage >= .95).to(torch.float32)
    specificity = 1. + .25 * torch.log1p(w['comparison_size'])
    rule = coverage.pow(6.) * identified * (positive - negative + implicit) * specificity[None, :]
    # Bind a nearby explicit scalar attribute after each comparative cue.
    # An unavailable requested work value must not fall back to ranking a
    # merely mentioned force. Conflicting requested attributes abstain.
    positions = torch.arange(q.shape[1], device=q.device)[None, :]
    attributes = (w['token_raw_concept'][q][:, :, None] == w['comparison_attribute_ids'][None, None, :]).to(torch.float32)
    following = (positions[:, None, :] > positions[:, :, None]) & (positions[:, None, :] <= positions[:, :, None] + 4)
    eligible = following & (torch.amax(attributes, dim=2)[:, None, :] > 0) & (request_scope[:, None, :] > 0)
    nearest = torch.amin(torch.where(eligible, positions[:, None, :], q.shape[1]), dim=2)
    picked = torch.nn.functional.pad(attributes, (0, 0, 0, 1)).gather(1, nearest[:, :, None].expand(-1, -1, attributes.shape[2]))
    picked = torch.amax(picked * (comparative * cue_scope)[:, :, None], dim=1)
    requested_property = (picked @ w['comparison_attribute_map']).clamp(0., 1.)
    requested_count = requested_property.sum(dim=1, keepdim=True)
    selection = (picked @ w['comparison_attribute_rules']).clamp(0., 1.) * (requested_count == 1).to(torch.float32)
    selected_rules = torch.where(requested_count > 0, selection, torch.ones_like(selection))
    # Filter rule contexts, not their output property: at fixed volume a
    # requested mass comparison legitimately uses the authored density rule.
    rule = rule * selected_rules
    property_positive = torch.zeros((batch, w['property_numeric'].shape[0]), device=q.device)
    property_negative = torch.zeros_like(property_positive)
    destinations = w['comparison_property'][None, :].expand(batch, -1)
    property_positive.scatter_reduce_(1, destinations, rule.clamp(min=0), reduce='amax', include_self=True)
    property_negative.scatter_reduce_(1, destinations, (-rule).clamp(min=0), reduce='amax', include_self=True)
    logical_not = torch.amax(w['comparison_not'][q] * negation_mask(w, q) * request_scope, dim=1)
    least_likely = torch.amax(probability[:, :, 2] * next_likely * request_scope, dim=1)
    strength = (property_positive - property_negative) * complete * ((1. - 2. * logical_not) * (1. - 2. * least_likely))[:, None]

    # Rank only comparable, known values; preserve equal-valued choices.
    difference = values[:, :, None, :] - values[:, None, :, :]
    tolerance = 1e-5 * torch.maximum(torch.abs(values[:, :, None, :]), torch.abs(values[:, None, :, :])).clamp(min=1e-8)
    ordering = (difference > tolerance).to(torch.float32) - (difference < -tolerance).to(torch.float32)
    rank = ordering.sum(dim=2) / 3.
    strongest = torch.amax(torch.abs(strength), dim=1, keepdim=True)
    selected = ((torch.abs(strength) >= strongest - 1e-6) & (torch.abs(strength) > 1e-6)).to(torch.float32)
    return 2.0 * (rank * (strength * selected)[:, None, :]).sum(dim=2) / selected.sum(dim=1, keepdim=True).clamp(min=1.)


def opposed(w, features, positions):
    # Literal opposites suppress a match only when the required predicate is
    # absent. A sentence containing both directions is not globally negated.
    literal = (positions < 1000.).to(torch.float32) * (features >= .90).to(torch.float32)
    alternative = torch.zeros_like(features)
    destination = w['opposite_target'][None, :].expand(features.shape[0], -1)
    alternative.scatter_reduce_(1, destination, literal[:, w['opposite_source']], reduce='amax', include_self=True)
    return alternative * (1. - literal)


def local_affirmation(w, tokens):
    negative = w['oneg'][tokens] * negation_mask(w, tokens)
    previous = torch.nn.functional.pad(negative[:, :-1], (3, 0)).unfold(1, 3, 1)
    return 1. - torch.amax(previous, dim=2)


def change_markers(w, tokens):
    # 'Does not increase' means absence of a proven increase, not a proven
    # decrease. Conservatively abstain on negated local change cues.
    return w['delta_tokens'][tokens] * local_affirmation(w, tokens)[:, :, None]


def qualitative(w, q, o, qf, qp, of, op):
    # Bind a local comparative marker to a named source quantity. A cue nearer
    # the responding quantity cannot also describe the source of this rule.
    qpos = torch.arange(q.shape[1], device=q.device)[None, :, None]
    source = qp[:, w['causal_source']]
    target = qp[:, w['causal_target']]
    # The final explicit target mention wins. Otherwise a component such as
    # 'effort' inside 'effort arm' would shadow the quantity actually asked for.
    raw_windows = torch.nn.functional.pad(w['token_raw_concept'][q], (0, 2)).unfold(1, 3, 1)
    target_matches = ((raw_windows[:, :, None, :] == w['causal_target_tokens'][None, None, :, :]) | (~w['causal_target_mask'][None, None, :, :])).to(torch.float32).prod(dim=3)
    last_target = torch.amax(torch.where(target_matches > 0, qpos, -1), dim=1)
    target = torch.where(last_target >= 0, last_target, target)
    displacement = qpos - source[:, None, :]
    source_radius = (w['causal_source_length'] - 1.) / 2.
    source_distance = (torch.abs(displacement - source_radius[None, None, :]) - source_radius[None, None, :]).clamp(min=0.)
    target_radius = (w['causal_target_length'] - 1.) / 2.
    target_distance = (torch.abs(qpos - target[:, None, :] - target_radius[None, None, :]) - target_radius[None, None, :]).clamp(min=0.)
    attribute_positions = qp[:, w['causal_attributes']]
    attribute_present = qf[:, w['causal_attributes']] >= .90
    attribute_radius = (w['causal_attribute_length'] - 1.) / 2.
    attribute_distance = (torch.abs(qpos - attribute_positions[:, None, :] - attribute_radius[None, None, :]) - attribute_radius[None, None, :]).clamp(min=0.)
    nearest_quantity = torch.amin(torch.where(attribute_present[:, None, :], attribute_distance, 1000.), dim=2)
    locality = torch.exp(-.35 * source_distance)
    locality = locality * ((displacement >= -3) & (source_distance <= 4) & (source_distance < target_distance) & (source_distance <= nearest_quantity[:, :, None]) & (source[:, None, :] < 1000.)).to(torch.float32)
    qmarker = change_markers(w, q)
    qpositive = torch.amax(locality * qmarker[:, :, 0, None], dim=1)
    qnegative = torch.amax(locality * qmarker[:, :, 1, None], dim=1)
    qdirection = torch.sign(qpositive - qnegative)
    unambiguous = ((qpositive * qnegative) == 0).to(torch.float32)
    changed = (torch.maximum(qpositive, qnegative) >= .24).to(torch.float32) * unambiguous
    expected = qdirection * w['causal_sign'][None, :]
    context = (qf @ w['causal_context'] + w['causal_bias'][None, :]).clamp(0., 1.)
    gate = changed * (context >= .95).to(torch.float32) * (qf[:, w['causal_source']] >= .90).to(torch.float32)
    # Explicitly fixed quantities override a generic all-else-equal law. Use
    # the first mention here, since the final mention may just ask about it.
    first_target_distance = (torch.abs(qpos - qp[:, w['causal_target']][:, None, :] - target_radius[None, None, :]) - target_radius[None, None, :]).clamp(min=0.)
    fixed_marker = (w['constant_tokens'][q] * local_affirmation(w, q))[:, :, None]
    source_fixed = torch.amax(fixed_marker * ((source_distance <= 2) & (source_distance <= nearest_quantity[:, :, None])).to(torch.float32), dim=1)
    target_fixed = torch.amax(fixed_marker * ((first_target_distance <= 2) & (first_target_distance <= nearest_quantity[:, :, None])).to(torch.float32), dim=1)
    gate = gate * (1. - source_fixed) * (1. - target_fixed)

    opos = torch.arange(o.shape[2], device=o.device)[None, None, :, None]
    optarget = op[:, :, w['causal_target']]
    odistance = opos - optarget[:, :, None, :]
    inside_name = (odistance >= 0) & (odistance < w['causal_target_length'][None, None, None, :]) & (w['causal_target_length'][None, None, None, :] > 1)
    olocality = torch.exp(-.35 * torch.abs(odistance)) * ((odistance >= -3) & (odistance <= 4) & (optarget[:, :, None, :] < 1000.) & (~inside_name)).to(torch.float32)
    omarker = change_markers(w, o.reshape(-1, o.shape[-1])).reshape(o.shape[0], 4, o.shape[-1], 2)
    oposit = torch.amax(olocality * omarker[:, :, :, 0, None], dim=2)
    onegat = torch.amax(olocality * omarker[:, :, :, 1, None], dim=2)
    # Short answers may omit a target explicitly named in the question, but
    # an option naming a different quantity must not borrow that target.
    named_attribute = torch.amax(of[:, :, w['causal_attributes']], dim=2)
    implicit = ((named_attribute < .90) & ((o > 0).sum(dim=2) <= 12)).to(torch.float32)
    global_positive = torch.amax(omarker[:, :, :, 0], dim=2)
    global_negative = torch.amax(omarker[:, :, :, 1], dim=2)
    target_in_question = (qf[:, w['causal_target']] >= .90).to(torch.float32)
    implicit_gate = implicit[:, :, None] * target_in_question[:, None, :]
    oposit = torch.maximum(oposit, global_positive[:, :, None] * implicit_gate)
    onegat = torch.maximum(onegat, global_negative[:, :, None] * implicit_gate)
    odirection = torch.sign(oposit - onegat)
    oclear = ((oposit * onegat) == 0).to(torch.float32)
    ogate = (torch.maximum(oposit, onegat) >= .24).to(torch.float32) * oclear
    target_available = torch.maximum((of[:, :, w['causal_target']] >= .90).to(torch.float32), implicit_gate)
    evidence = expected[:, None, :] * odirection * gate[:, None, :] * ogate * target_available
    return 1.35 * (torch.amax(evidence.clamp(min=0), dim=2) - torch.amax((-evidence).clamp(min=0), dim=2))


def named_interactions(w, q, o, qf, of):
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
    # Inverse reasoning identifies one counterpart, not a whole pair. In
    # particular, a pair of south poles is not one south pole being compared
    # to a north pole mentioned in an earlier narrative sentence.
    raw = w['token_raw_concept'][o]
    option_count = (raw[:, :, :, None] == w['interaction_ids'][None, None, None, 2:6]).to(torch.float32).sum(dim=2).sum(dim=2)
    multiple = torch.amax(((w['numeric'][o] >= 2) | (raw == w['interaction_ids'][7]) | (raw == w['interaction_ids'][13])).to(torch.float32), dim=2)
    single = (option_count == 1).to(torch.float32) * (1. - multiple)
    inverse = magnetic_target[:, None] * (io[:, :, 2] - io[:, :, 3]) + electric_target[:, None] * (io[:, :, 4] - io[:, :, 5])
    return 1.8 * (interaction + inverse * single)


def pair_choices(w, q, o, qfeatures, recent):
    # Answer choices can describe the pairs instead of naming the interaction.
    # Count literal identities: a bag containing 'north' cannot distinguish
    # one north pole from two north poles. No unspecified object is assigned
    # a charge or magnetic polarity.
    qraw = w['token_raw_concept'][q]
    request_scope = answer_scope(w, q, recent)
    qliteral = torch.amax((qraw[:, :, None] == w['interaction_ids'][None, None, :]).to(torch.float32) * request_scope[:, :, None], dim=1)
    pair_request = torch.maximum(torch.amax(qliteral[:, 6:8], dim=1), qliteral[:, 13])
    action = qliteral[:, 11] - qliteral[:, 10]
    domain = torch.stack((torch.maximum(qfeatures[:, w['interaction_ids'][0]], qfeatures[:, w['interaction_ids'][12]]), qfeatures[:, w['interaction_ids'][1]]), dim=1)
    domain = (domain >= .99).to(torch.float32)
    request_gate = pair_request * (domain.sum(dim=1) == 1).to(torch.float32)

    raw = w['token_raw_concept'][o]
    identities = (raw[:, :, :, None] == w['interaction_ids'][None, None, None, 2:6]).to(torch.float32).sum(dim=2).reshape(q.shape[0], 4, 2, 2)
    total = identities.sum(dim=3)
    numeric_count = (w['numeric'][o] >= 0).sum(dim=2)
    two_count = (w['numeric'][o] == 2).sum(dim=2)
    both = torch.amax((raw == w['interaction_ids'][7]).to(torch.float32), dim=2)
    explicit_pair = (total == 2) & (numeric_count[:, :, None] == 0)
    repeated_pair = (total == 1) & ((((numeric_count == 1) & (two_count == 1)) | ((numeric_count == 0) & (both > 0)))[:, :, None])
    no_mixed_domain = (total.sum(dim=2, keepdim=True) - total) == 0
    valid = (explicit_pair | repeated_pair) & no_mixed_domain
    like = (torch.amin(identities, dim=3) == 0).to(torch.float32)
    relation = 2. * like - 1.
    qualifiers = torch.amax((raw[:, :, :, None] == w['interaction_ids'][None, None, None, 8:10]).to(torch.float32), dim=2)
    consistent = (qualifiers[:, :, 0:1] * (1. - like) + qualifiers[:, :, 1:2] * like) == 0
    evidence = (relation * valid.to(torch.float32) * consistent.to(torch.float32) * domain[:, None, :]).sum(dim=2)
    return 1.8 * evidence * (action * request_gate)[:, None] * (1. - option_negation(w, o))


def experimental_roles(w, q, o, start, recent, question_negative):
    # Bind explicit verbs to short noun spans, only in the narrative preceding
    # a final question that explicitly names one experimental-variable role.
    p = torch.arange(q.shape[1], device=q.device)[None, :]
    separation = p[:, None, :] - p[:, :, None]
    request_scope = answer_scope(w, q, recent)
    variable_near = torch.amax(w['experiment_variable'][q][:, None, :] * request_scope[:, None, :] * (torch.abs(separation) <= 3).to(torch.float32), dim=2)
    requested = torch.amax(w['experiment_request'][q] * (request_scope * variable_near)[:, :, None], dim=1)
    request_gate = (requested.sum(dim=1) == 1).to(torch.float32) * (start[:, 0] > 0).to(torch.float32) * (1. - question_negative)

    cue = w['experiment_role'][q] * ((p < start).to(torch.float32) * local_affirmation(w, q))[:, :, None]
    preceding_copula = torch.nn.functional.pad(w['experiment_copula'][q[:, :-1]], (1, 0))
    earlier_copula = torch.nn.functional.pad(w['experiment_copula'][q[:, :-2]], (2, 0))
    copula_distance = torch.where(preceding_copula > 0, 1, torch.where(earlier_copula > 0, 2, 0))
    passive = (w['experiment_passive'][q] > 0) & (copula_distance > 0)
    anchor = p - copula_distance * passive.to(torch.int64)
    following_block = torch.nn.functional.pad(w['experiment_blocked_follow'][q[:, 1:]], (0, 1))
    cue = cue * (passive | (following_block == 0)).to(torch.float32)[:, :, None]
    boundary = torch.maximum(w['experiment_boundary'][q], torch.amax(w['experiment_role'][q], dim=2)) > 0
    left_boundary = torch.amax(torch.where(boundary[:, None, :] & (p[:, None, :] < anchor[:, :, None]), p[:, None, :], -1), dim=2)
    right_boundary = torch.amin(torch.where(boundary[:, None, :] & (p[:, None, :] > anchor[:, :, None]), p[:, None, :], q.shape[1]), dim=2)
    offset = p[:, None, :] - anchor[:, :, None]
    active_span = (~passive[:, :, None]) & (offset > 0) & (offset <= 8)
    passive_span = passive[:, :, None] & (offset < 0) & (offset >= -8)
    span = (active_span | passive_span) & (p[:, None, :] > left_boundary[:, :, None]) & (p[:, None, :] < right_boundary[:, :, None]) & (p[:, None, :] < start[:, :, None]) & (q[:, None, :] > 1)
    # Keeping/holding is not evidence of a constant unless explicitly qualified.
    next_boundary = torch.amin(torch.where(boundary[:, None, :] & (separation > 0), p[:, None, :], q.shape[1]), dim=2)
    fixed = torch.amax((w['experiment_constant'][q] * local_affirmation(w, q))[:, None, :] * ((separation > 0) & (separation <= 8) & (p[:, None, :] < next_boundary[:, :, None]) & (p[:, None, :] < start[:, :, None])).to(torch.float32), dim=2)
    cue = torch.cat((cue[:, :, :2], cue[:, :, 2:] * fixed[:, :, None]), dim=2)
    cue = cue * (span.to(torch.float32).sum(dim=2) > 0).to(torch.float32)[:, :, None]
    role_count = cue.sum(dim=1)
    # Multiple descriptions of one role are deliberately not merged into a
    # fabricated conjunction. Unsupported or ambiguous bindings abstain.
    bound = (cue.transpose(1, 2) @ span.to(torch.float32)).clamp(0., 1.)
    bound = bound * (role_count == 1).to(torch.float32)[:, :, None]
    qcontent = (1. - w['experiment_ignore'][q]) * (q > 1).to(torch.float32)
    bound = bound * qcontent[:, None, :]
    ocontent = (1. - w['experiment_ignore'][o]) * (o > 1).to(torch.float32)
    match = (w['token_concept'][o][:, :, :, None] == w['token_concept'][q][:, None, None, :]).to(torch.float32)
    covered = (match @ bound.transpose(1, 2)[:, None, :, :]).clamp(0., 1.)
    option_mass = ocontent.sum(dim=2)
    coverage = (covered * ocontent[:, :, :, None]).sum(dim=2) / option_mass[:, :, None].clamp(min=1.)
    # A material alone cannot stand for a property of that material:
    # 'water' is not a match for 'temperature of water'.
    matched_q = torch.amax(match * ocontent[:, :, :, None], dim=2)
    missing_head = ((1. - matched_q)[:, :, None, :] * bound[:, None, :, :] * w['experiment_head'][q][:, None, None, :]).sum(dim=3)
    valid_option = (option_mass > 0) & ((o == 1).sum(dim=2) == 0) & (option_negation(w, o) == 0)
    role_match = ((coverage >= .999) & (missing_head == 0) & valid_option[:, :, None]).to(torch.float32)
    unambiguous = (role_match.sum(dim=2) == 1).to(torch.float32)
    support = (role_match * requested[:, None, :]).sum(dim=2)
    conflict = (role_match * (1. - requested[:, None, :])).sum(dim=2)
    return 1.35 * (support - conflict) * unambiguous * request_gate[:, None]


def trophic_effects(w, q, o, start, recent):
    # Extract only explicit short 'A eats B' / 'B is eaten by A' clauses.
    # Each token position is a possible verb; all binding is parallel tensor work.
    raw = w['token_raw_concept'][q]
    p = torch.arange(q.shape[1], device=q.device)[None, :]
    distance = p[:, None, :] - p[:, :, None]
    entity = w['ecology_entity'][q] > 0
    boundary = w['ecology_boundary'][q] > 0
    left_boundary = torch.amax(torch.where(boundary[:, None, :] & (distance < 0), p[:, None, :], -1), dim=2)
    right_boundary = torch.amin(torch.where(boundary[:, None, :] & (distance > 0), p[:, None, :], 1000), dim=2)
    left = torch.amax(torch.where(entity[:, None, :] & (distance < 0) & (distance >= -5) & (p[:, None, :] > left_boundary[:, :, None]), p[:, None, :], -1), dim=2)
    right = torch.amin(torch.where(entity[:, None, :] & (distance > 0) & (distance <= 5) & (p[:, None, :] < right_boundary[:, :, None]), p[:, None, :], 1000), dim=2)
    left_id = raw.gather(1, left.clamp(0, q.shape[1] - 1))
    right_id = raw.gather(1, right.clamp(0, q.shape[1] - 1))
    next_on = torch.nn.functional.pad(w['ecology_on'][q[:, 1:]], (0, 1))
    next_by = torch.nn.functional.pad(w['ecology_by'][q[:, 1:]], (0, 1))
    passive = w['ecology_passive'][q] * next_by
    active = torch.maximum(w['ecology_active'][q], w['ecology_active_on'][q] * next_on)
    valid_edge = ((left >= 0) & (right < 1000) & (left_id != right_id)).to(torch.float32) * torch.maximum(active, passive) * local_affirmation(w, q)
    consumer = torch.where(passive > 0, right_id, left_id)
    food = torch.where(passive > 0, left_id, right_id)

    # A population change is a premise only before the final request, or in
    # an explicit conditional clause. Outcome cues in the request are excluded.
    condition = torch.amax(torch.where(w['ecology_conditional'][q] > 0, p, -1000), dim=1, keepdim=True)
    conditional_premise = (p > condition) & (p - condition <= 12) & ((start < 0) | (condition > start) | (p < start))
    premise = ((p < start) | conditional_premise).to(torch.float32)
    marker_up = w['ecology_up'][q] * local_affirmation(w, q)
    marker_down = w['ecology_down'][q] * local_affirmation(w, q)
    nearest_distance = torch.amin(torch.where(entity[:, None, :], torch.abs(distance), 1000), dim=2)
    nearest = entity[:, None, :] & (torch.abs(distance) == nearest_distance[:, :, None])
    nearest_index = torch.argmax(nearest.to(torch.float32), dim=2)
    nearest_id = raw.gather(1, nearest_index)
    conflicting = torch.amax((nearest & (raw[:, None, :] != nearest_id[:, :, None])).to(torch.float32), dim=2)
    # Do not turn changes in body mass, temperature, food supply, or an
    # unparsed coordinated subject into a change in species abundance.
    local_clause = (torch.abs(distance) <= 4) & (p[:, None, :] > left_boundary[:, :, None]) & (p[:, None, :] < right_boundary[:, :, None])
    competing_attribute = torch.amax(w['ecology_source_block'][q][:, None, :] * local_clause.to(torch.float32), dim=2)
    bridge_left = torch.minimum(nearest_index, p)
    bridge_right = torch.maximum(nearest_index, p)
    bridge = (p[:, None, :] > bridge_left[:, :, None]) & (p[:, None, :] < bridge_right[:, :, None])
    unsupported_bridge = torch.amax(bridge.to(torch.float32) * (1. - w['ecology_source_bridge'][q][:, None, :]), dim=2)
    marker_gate = premise * (nearest_distance <= 4).to(torch.float32) * (1. - conflicting) * (1. - competing_attribute) * (1. - unsupported_bridge)
    source_up = torch.zeros((q.shape[0], w['feature_weight'].shape[0]), device=q.device)
    source_down = torch.zeros_like(source_up)
    source_up.scatter_reduce_(1, nearest_id, marker_up * marker_gate, reduce='amax', include_self=True)
    source_down.scatter_reduce_(1, nearest_id, marker_down * marker_gate, reduce='amax', include_self=True)
    changed = source_up - source_down
    unique_source = ((source_up + source_down > 0).sum(dim=1) == 1).to(torch.float32) * ((source_up * source_down).sum(dim=1) == 0).to(torch.float32)

    # More food supports its consumer; more consumers put pressure on food.
    # This is direct, ceteris-paribus reasoning, not an unrestricted simulation.
    consumer_effect = changed.gather(1, food) * valid_edge
    food_effect = -changed.gather(1, consumer) * valid_edge
    positive = torch.zeros_like(changed)
    negative = torch.zeros_like(changed)
    positive.scatter_reduce_(1, consumer, consumer_effect.clamp(min=0), reduce='amax', include_self=True)
    positive.scatter_reduce_(1, food, food_effect.clamp(min=0), reduce='amax', include_self=True)
    negative.scatter_reduce_(1, consumer, (-consumer_effect).clamp(min=0), reduce='amax', include_self=True)
    negative.scatter_reduce_(1, food, (-food_effect).clamp(min=0), reduce='amax', include_self=True)
    expected = (positive - negative) * (1. - (source_up + source_down).clamp(0., 1.))

    request_scope = answer_scope(w, q, recent)
    # A trailing if/when clause supplies premises, not the requested target or
    # attribute. Preserve it for source binding but exclude it from the goal.
    trailing_condition = torch.amin(torch.where((w['ecology_conditional'][q] > 0) & (p > start), p, 1000), dim=1, keepdim=True)
    request_scope = request_scope * (p < trailing_condition).to(torch.float32)
    requested = torch.zeros_like(changed)
    requested.scatter_reduce_(1, raw, w['ecology_entity'][q] * request_scope * (1. - (source_up + source_down).clamp(0., 1.)).gather(1, raw), reduce='amax', include_self=True)
    unique_requested = (requested.sum(dim=1) == 1).to(torch.float32)
    implied = (expected * requested).sum(dim=1) * unique_requested
    oraw = w['token_raw_concept'][o]
    named = torch.zeros((q.shape[0], 4, w['feature_weight'].shape[0]), device=q.device)
    named.scatter_reduce_(2, oraw, w['ecology_entity'][o], reduce='amax', include_self=True)
    count = named.sum(dim=2)
    named_effect = (named * expected[:, None, :]).sum(dim=2)
    # A true statement about another organism is not an answer about the
    # organism explicitly requested. Bare direction options still need the
    # unique-target binding above.
    target_agreement = ((named * requested[:, None, :]).sum(dim=2) > 0) | (requested.sum(dim=1, keepdim=True) == 0)
    named_effect = named_effect * target_agreement.to(torch.float32)
    effect = torch.where(count == 0, implied[:, None], named_effect * (count == 1).to(torch.float32))
    oup = torch.amax(w['ecology_up'][o], dim=2)
    odown = torch.amax(w['ecology_output_down'][o], dim=2)
    direction = (oup - odown) * (1. - oup * odown)
    # Borrow a target only in a genuinely simple population-change answer.
    # Unknown names, other properties, quantities, extinction claims, and
    # additional clauses are outside this qualitative module's entailments.
    allowed = torch.maximum(w['ecology_output_words'][o], w['ecology_entity'][o])
    allowed = torch.maximum(allowed, torch.maximum(w['ecology_up'][o], w['ecology_output_down'][o]))
    valid_option = ((allowed > 0) | (o == 0)).to(torch.float32).prod(dim=2)
    request = torch.amax(w['ecology_request'][q] * request_scope, dim=1)
    unsupported_request = torch.amax(torch.maximum(w['ecology_source_block'][q], (q == 1).to(torch.float32)) * request_scope, dim=1)
    return 1.65 * effect * direction * valid_option * (unique_source * request * (1. - unsupported_request))[:, None] * (1. - option_negation(w, o))


def separation_methods(w, q, o, of, op, start, recent, negative):
    # Recognize all material occurrences, including repeated goal mentions.
    p = torch.arange(q.shape[1], device=q.device)[None, :]
    windows = torch.nn.functional.pad(w['token_raw_concept'][q], (0, 2)).unfold(1, 3, 1)
    mask = w['separation_material_mask']
    matched = ((windows[:, :, None, :] == w['separation_material_tokens'][None, None, :, :]) | (~mask[None, None, :, :])).to(torch.float32).prod(dim=3)
    multiple = (mask.sum(dim=1) > 1).to(torch.float32)
    consumed = torch.zeros_like(q, dtype=torch.float32)
    for offset in range(3):
        covered = torch.amax(matched * multiple[None, None, :] * mask[None, None, :, offset], dim=2)
        consumed = torch.maximum(consumed, torch.nn.functional.pad(covered, (offset, 0))[:, :q.shape[1]])
    matched = matched * (1. - consumed[:, :, None] * (1. - multiple[None, None, :]))
    constituents = torch.amax(matched @ w['separation_material_components'], dim=1).clamp(0., 1.)
    targets = (matched @ w['separation_material_target']).clamp(0., 1.)

    separate = w['separation_separate'][q]
    recover = w['separation_recover'][q]
    remove = w['separation_remove'][q]
    actions = torch.maximum(separate, torch.maximum(recover, remove))
    request = answer_scope(w, q, recent)
    selection = torch.amax(w['separation_select'][q] * (p == start).to(torch.float32), dim=1)
    local_action = torch.amax(actions * request, dim=1)
    method_request = torch.amax(w['separation_method'][q] * request, dim=1)
    competing_task = torch.amax(w['separation_other_task'][q] * request, dim=1)
    request_block = torch.amax(w['separation_request_block'][q] * request, dim=1)
    context_block = torch.amax(w['separation_context_block'][q], dim=1)
    excluded = (w['separation_material_components'].sum(dim=1) == 0).to(torch.float32)
    context_block = torch.maximum(context_block, torch.amax(matched * excluded[None, None, :], dim=(1, 2)))
    unsupported = torch.amax((q == 1).to(torch.float32), dim=1)
    negated = torch.maximum(negative, option_negation(w, q))
    gate = selection * torch.maximum(local_action, method_request) * torch.amax(actions, dim=1)
    gate = gate * (1. - competing_task) * (1. - request_block) * (1. - context_block) * (1. - unsupported) * (1. - negated)
    gate = gate * (constituents.sum(dim=1) == 2).to(torch.float32)

    # A short object before 'from' is a product, not the source mixture.
    # A mixture head or another action stops the binding span. Generic
    # 'separate salt and water' intentionally has no singled-out product.
    later = p[:, None, :] > p[:, :, None]
    boundary_words = torch.maximum(w['separation_from'][q], torch.maximum(w['separation_mixture_head'][q], actions))
    boundary = torch.amin(torch.where(later & (boundary_words[:, None, :] > 0), p[:, None, :], 1000), dim=2)
    span = later & (p[:, None, :] <= p[:, :, None] + 8) & (p[:, None, :] < boundary[:, :, None])
    span = span.to(torch.float32) * actions[:, :, None]
    raw_target = (span @ targets).clamp(0., 1.)
    single = (raw_target.sum(dim=2) == 1).to(torch.float32)
    bad_prefix = (1. - w['separation_bridge'][q]).cumsum(dim=1)
    before = torch.nn.functional.pad(bad_prefix[:, :-1], (1, 0))
    clear_bridge = (before[:, None, :] - bad_prefix[:, :, None] == 0).to(torch.float32)
    bound = ((span * clear_bridge) @ targets).clamp(0., 1.) * single[:, :, None]
    bound_valid = (bound.sum(dim=2) == 1).to(torch.float32)
    invalid_goal = torch.amax(torch.maximum(recover, remove) * (1. - bound_valid), dim=1)
    desired = torch.where(remove[:, :, None] > 0, (constituents[:, None, :] - bound).clamp(0., 1.), bound)
    desired = torch.amax(desired * bound_valid[:, :, None], dim=1)
    gate = gate * (1. - invalid_goal) * (desired.sum(dim=1) <= 1).to(torch.float32)

    # Specific product rows supersede wildcard rows for the same binary pair.
    pair_match = (constituents @ w['separation_rule_components'].transpose(0, 1) == 2).to(torch.float32)
    specific = pair_match * (desired @ w['separation_rule_target'].transpose(0, 1))
    fallback = (w['separation_rule_target'].sum(dim=1) == 0).to(torch.float32)
    selected = specific + pair_match * fallback[None, :] * (1. - torch.amax(specific, dim=1, keepdim=True))
    dissolved = torch.amax(w['separation_dissolved'][q], dim=1)
    selected = selected * (1. - dissolved[:, None] * w['separation_rule_dissolved'][None, :])
    method_scores = selected @ w['separation_rule_methods']

    # Only one literal named method; arbitrary procedural claims remain with
    # retrieval. Unknown words, extra clauses and negation cannot disappear.
    method = (((of >= .99) & (op < 1000.)).to(torch.float32) @ w['separation_method_map']).clamp(0., 1.)
    allowed = ((w['separation_option_allowed'][o] > 0) | (o == 0)).to(torch.float32).prod(dim=2)
    valid = allowed * (method.sum(dim=2) == 1).to(torch.float32) * (1. - option_negation(w, o))
    return w['separation_scale'] * (method * method_scores[:, None, :]).sum(dim=2) * valid * gate[:, None]


def forward(w, q, o):
    qf, qp = concepts(w, q)
    option_features, option_positions = concepts(w, o.reshape(-1, o.shape[-1]))
    of = option_features.reshape(q.shape[0], 4, -1)
    op = option_positions.reshape(q.shape[0], 4, -1)

    # Ontological membership is weaker question context than an explicit word:
    # merely mentioning water must not act like asking about all compounds.
    # Keep full inheritance for option typing and the symbolic modules.
    retrieval_qf = qf * torch.where(qp < 1000., 1., .5)
    # Context coverage has a steep gate: several mutually consistent clues
    # outrank a solitary word in a broad topic.
    coverage = (project(retrieval_qf, w['qmat'])).clamp(0., 1.)
    query_conflict = (project(opposed(w, qf, qp), w['qmat'])).clamp(0., 1.)
    activation = coverage.pow(6.) * (1. - query_conflict).pow(2.) * (0.70 + 0.30 * torch.sqrt(w['qmass']))[None, :]
    # Prefer well-covered contexts without turning weak evidence into certainty.
    retrieval_priority = activation * w['context_confidence'][None, :]
    strongest = torch.amax(retrieval_priority, dim=1, keepdim=True)
    activation = activation * (0.40 + 0.60 * (retrieval_priority / strongest.clamp(min=1e-6)).pow(2.))
    context = activation[:, w['acontext']] * w['confidence'][None, :]

    # Candidate words shared by every option or already supplied by the
    # question provide less discriminating evidence than novel answer content.
    distinct = (1.0 - 0.70 * of.mean(dim=1)) * (1.0 - 0.45 * qf)
    raw_coverage = (project(of, w['amat'])).clamp(0., 1.)
    distinctive_coverage = (project(of * distinct[:, None, :], w['amat'])).clamp(0., 1.)
    option_conflict = (project(opposed(w, option_features, option_positions), w['amat'])).reshape(q.shape[0], 4, -1).clamp(0., 1.)
    evidence = raw_coverage.pow(2.5) * (1. - option_conflict).pow(3.) * (0.30 + 0.70 * distinctive_coverage / raw_coverage.clamp(min=1e-6))
    # Match precision complements answer recall: appending unrelated content
    # must not increase confidence in a completion. Unknown tokens carry
    # uncertainty mass rather than silently disappearing from precision.
    unknown_mass = 2. * (o == 1).to(torch.float32).sum(dim=2)
    distinctive_mass = (of * distinct[:, None, :] * w['feature_weight'][None, None, :]).sum(dim=2) + unknown_mass
    precision = distinctive_coverage * w['amass'][None, None, :] / distinctive_mass[:, :, None].clamp(min=1e-6)
    evidence = evidence * (.5 + .5 * torch.sqrt(precision.clamp(0., 1.)))
    evidence = evidence * (0.85 + 0.15 * torch.sqrt(w['amass']))[None, None, :]
    evidence = evidence * context[:, None, :]

    # An assertion about the absence of a property has the opposite polarity
    # to one about its presence. The polarity is also reversed by a negated
    # option, independently of whether the question asks for an exception.
    option_negative = option_negation(w, o)
    polarity = w['asign'][None, None, :] * w['apolarity'][None, None, :] * (1. - 2. * option_negative[:, :, None])
    positions, last_question_start, recent, question_negative = question_scope(w, q)
    direct = w['context_direct'][w['acontext']]
    question_sign = 1. - 2. * question_negative[:, None] * (1. - direct[None, :])
    polarity = polarity * question_sign[:, None, :]

    positive = evidence * (polarity > 0).to(torch.float32)
    negative = evidence * (polarity < 0).to(torch.float32)
    support = (torch.topk(positive, 4, dim=2).values * w['top_weights']).sum(dim=2)
    contradiction = (torch.topk(negative, 4, dim=2).values * w['top_weights']).sum(dim=2)
    logits = support - 0.85 * contradiction

    # Whole-statement completion is a secondary, undirected backoff. It lets
    # clues occur on either side of an authored relation, without fabricating
    # new training examples or treating a single shared topic as entailment.
    # Only option features absent from the question can add coverage.
    cmass = w['qmass'][w['acontext']]
    total_mass = (cmass + w['amass']).clamp(min=1e-6)
    qanswer = (project(retrieval_qf, w['amat'])).clamp(0., 1.)
    question_fact = (coverage[:, w['acontext']] * cmass[None, :] + qanswer * w['amass'][None, :]) / total_mass[None, :]
    novel = of * (1. - qf[:, None, :])
    novel_context = project(novel, w['qmat'])
    novel_answer = project(novel, w['amat'])
    added_coverage = (novel_context[:, :, w['acontext']] * cmass[None, None, :] + novel_answer * w['amass'][None, None, :]) / total_mass[None, None, :]
    full_coverage = (question_fact[:, None, :] + added_coverage).clamp(0., 1.)
    option_context = project(of * distinct[:, None, :], w['qmat'])
    option_mass = ((of * w['feature_weight'][None, None, :]).sum(dim=2) + unknown_mass).clamp(min=1e-6)
    answer_precision = (option_context[:, :, w['acontext']] * cmass[None, None, :] + distinctive_coverage * w['amass'][None, None, :]) / option_mass[:, :, None]
    completion = full_coverage.pow(6.) * question_fact[:, None, :].pow(2.) * answer_precision.clamp(0., 1.)
    # A false completion cannot be contraposed into a negative definition of
    # its subject. Only affirmative facts participate in undirected backoff.
    affirmative_fact = ((w['asign'] > 0) & (w['apolarity'] > 0)).to(torch.float32)
    completion = completion * w['confidence'][None, None, :] * affirmative_fact[None, None, :] * (1. - direct[None, None, :]) * (1. - option_conflict).pow(2.)
    statement_support = torch.amax(completion * (polarity > 0).to(torch.float32), dim=2)
    statement_conflict = torch.amax(completion * (polarity < 0).to(torch.float32), dim=2)
    logits = logits + .90 * (statement_support - .85 * statement_conflict)
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

    # Equality and opposition require identity and multiplicity, rather than
    # merely a bag of words; inverse counterpart lookup is single-object only.
    logits = logits + named_interactions(w, q, o, qf, of) * (1. - 2. * question_negative[:, None]) * (1. - 2. * option_negative)

    # The first explicitly named class in the final question is a modest
    # category constraint. Inferred classes are never treated as question text.
    class_positions = qp[:, w['tax_classes']]
    class_prompt = torch.amax(w['class_question'][q] * (positions == last_question_start).to(torch.float32), dim=1, keepdim=True)
    eligible = (class_positions >= last_question_start) & (class_positions <= last_question_start + 10) & (class_positions < 1000.) & (class_prompt > 0)
    first_class = torch.amin(torch.where(eligible, class_positions, 1000.), dim=1, keepdim=True)
    requested = ((class_positions == first_class) & eligible).to(torch.float32)
    # 'Which animal is an arachnid?' requests arachnid, not merely animal.
    # Only a copular bridge without an intervening content verb may override
    # the first noun; 'which animal eats a plant' still requests animal.
    before_class = (positions[:, :, None] < class_positions[:, None, :]) & (positions[:, :, None] >= last_question_start[:, :, None])
    linking = (w['class_link'][q][:, :, None] > 0) & before_class
    link_position = torch.amax(torch.where(linking, positions[:, :, None], -1000), dim=1)
    between_link_class = (positions[:, :, None] > link_position[:, None, :]) & (positions[:, :, None] < class_positions[:, None, :])
    intervening_content = torch.amax(between_link_class.to(torch.float32) * (1. - w['class_bridge'][q][:, :, None]), dim=1)
    predicate_class = (class_positions < 1000.) & (link_position >= last_question_start) & (class_positions - link_position <= 5) & (intervening_content == 0) & (class_prompt > 0)
    first_predicate = torch.amin(torch.where(predicate_class, class_positions, 1000.), dim=1, keepdim=True)
    predicate_request = ((class_positions == first_predicate) & predicate_class).to(torch.float32)
    requested = torch.where(predicate_request.sum(dim=1, keepdim=True) > 0, predicate_request, requested)
    membership = (of[:, :, w['tax_classes']] * requested[:, None, :]).sum(dim=2) / requested.sum(dim=1, keepdim=True).clamp(min=1.)
    logits = logits + .75 * membership * (1. - 2. * question_negative[:, None]) * (1. - 2. * option_negative)
    logits = logits + comparisons(w, q, o, qf, op, recent)
    logits = logits + qualitative(w, q, o, qf, qp, of, op) * (1. - 2. * question_negative[:, None]) * (1. - 2. * option_negative)
    logits = logits + experimental_roles(w, q, o, last_question_start, recent, question_negative)
    logits = logits + pair_choices(w, q, o, qf, recent) * (1. - 2. * question_negative[:, None])
    logits = logits + trophic_effects(w, q, o, last_question_start, recent) * (1. - 2. * question_negative[:, None])
    logits = logits + separation_methods(w, q, o, of, op, last_question_start, recent, question_negative)
    return logits
