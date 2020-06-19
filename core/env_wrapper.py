import gym


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MultiStepWrapper, self).__init__(env)

    def step(self, action, action_repeat_n=1):
        assert action_repeat_n >= 1, 'action repeat should be atleast 1'

        reward_sum = 0
        # print(action_repeat_n)
        for i in range(action_repeat_n):
            obs, reward, done, info = super().step(action)
            reward_sum += reward
            if done:
                break

        return obs, reward_sum, done, info
