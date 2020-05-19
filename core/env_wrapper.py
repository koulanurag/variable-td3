import gym


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, env, gamma: float):
        super(MultiStepWrapper, self).__init__(env)

        self.gamma = gamma

    def step(self, action, action_repeat_n=1):
        assert action_repeat_n >= 1, 'action repeat should be atleast 1'

        reward_sum, steps = 0, 0
        for i in range(action_repeat_n):
            obs, reward, done, info = super().step(action)
            reward_sum += (self.gamma ** i) * reward
            steps += 1
            if done:
                break

        info['steps'] = steps

        return obs, reward_sum, done, info
