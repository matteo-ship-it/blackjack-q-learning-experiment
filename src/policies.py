STICK = 0
HIT = 1


def random_policy(obs, env):
    """Select a random action, ignoring the current observation."""
    return env.action_space.sample()


hard_standing_numbers = {
    1: 17,
    2: 13,
    3: 13,
    4: 12,
    5: 12,
    6: 12,
    7: 17,
    8: 17,
    9: 17,
    10: 17,
}

soft_standing_numbers = {
    1: 18,
    2: 18,
    3: 18,
    4: 18,
    5: 18,
    6: 18,
    7: 18,
    8: 18,
    9: 19,
    10: 19,
}


def reduced_basic_strategy(obs, env):
    """Select an action using a simplified Blackjack basic strategy."""
    player_sum, dealer_card, usable_ace = obs

    if usable_ace:
        minimum_standing_number = soft_standing_numbers[dealer_card]
    else:
        minimum_standing_number = hard_standing_numbers[dealer_card]

    if player_sum < minimum_standing_number:
        return HIT
    else:
        return STICK