import gym
import slimevolleygym

# Random policy class and rollout function taken from the slimevolley github page:
# https://github.com/hardmaru/slimevolleygym/blob/master/eval_agents.py

class RandomPolicy:
  def __init__(self):
    self.action_space = gym.spaces.MultiBinary(3)
    pass
  def predict(self, obs):
    return self.action_space.sample()

def rollout(env, policy0, policy1, render_mode=False):
  """ play one agent vs the other in modified gym-style loop. """
  obs0 = env.reset()
  obs1 = obs0 # same observation at the very beginning for the other agent

  done = False
  total_reward = 0
  #count = 0

  while not done:

    action0 = policy0.predict(obs0)
    action1 = policy1.predict(obs1)

    # uses a 2nd (optional) parameter for step to put in the other action
    # and returns the other observation in the 4th optional "info" param in gym's step()
    obs0, reward, done, info = env.step(action0, action1)
    obs1 = info['otherObs']

    total_reward += reward

    if render_mode:
      env.render()
      """ # used to render stuff to a gif later.
      img = env.render("rgb_array")
      filename = os.path.join("gif","daytime",str(count).zfill(8)+".png")
      cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
      count += 1
      """
      sleep(0.01)

  return total_reward

def make_slime_env():
    return gym.make("SlimeVolley-v0")

def evaluate_against_random(policy_func, render=False):
    random_policy = RandomPolicy()
    env = make_slime_env()
    reward = rollout(env, policy_func, random_policy, render_mode=render)
    print(f"Your agent reached a reward of {reward} playing against a random agent.")