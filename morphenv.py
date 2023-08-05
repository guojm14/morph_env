
import os
import gym
from gym.envs.registration import register
ENV_DIR = '/lustre/S/guojiaming/offlinerl/morph_pre/environments'
XML_DIR = '/lustre/S/guojiaming/offlinerl/morph_pre/environments/xmls'
def registerEnv(env_name, max_episode_steps=200, custom_xml=None):
    if not custom_xml:
        xml=os.path.join(XML_DIR, "{}.xml".format(env_name))
    # custom envs
    else:
        if os.path.isfile(custom_xml):
            xml=custom_xml
            
    # register each env

    env_name = os.path.basename(xml)[:-4]
    env_file = env_name
    params = {'xml': os.path.abspath(xml)}
    # register with gym
    register(id=("%s-v0" % env_name),
             max_episode_steps=max_episode_steps,
             entry_point="environments.%s:ModularEnv" % env_file,
             kwargs=params)

def makeEnv(name,seed=0):
    env=gym.make("environments:%s-v0" % name)
    env.seed(seed)
    return env

