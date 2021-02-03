FPS = 0.0001


def obs_to_state(env, obs):
    obs_i = env.state2str(obs)
    return env.states[obs_i]


def evaluate_agent(_env, _agent, _episode_count=200, render=False):
    rewards_test = []
    for i in range(_episode_count):
        obs = _env.reset()
        rsum = 0
        _env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if _env.verbose and render:
            _env.render(FPS)
        while True:
            if _env.verbose and render:
                _env.render(FPS)
            # Choose action
            state = obs_to_state(_env, obs)
            action = _agent.act(state)
            obs, reward, done, _ = _env.step(action)
            rsum += reward

            if done:
                rewards_test += [rsum]
                break
    return rewards_test