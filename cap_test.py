import time
import gym
import gym_cap
import numpy as np


# the modules that you can use to generate the policy.
import policy.roomba
import policy.random

start_time = time.time()
env = gym.make("cap-v0") # initialize the environment

done = False
t = 0
rscore = [0] * 20

# reset the environment and select the policies for each of the team
observation = env.reset(map_size=20,
                        render_mode="env",
                        policy_blue=policy.random.PolicyGen(env.get_map, env.get_team_blue),
                        policy_red=policy.roomba.PolicyGen(env.get_map, env.get_team_red))

overallGames = []
overallScores = []
individualTurns = []
individualTurnScores = []
turnCounter = 0
turnLimit = 50
gameLimit = 100


for x in range(0, gameLimit):
    individualTurns = []
    individualTurnScores = []
    turnCounter = 0
    while not done:

        #you are free to select a random action
        # or generate an action using the policy
        # or select an action manually
        # and the apply the selected action to blue team
        # or use the policy selected and provided in env.reset
        #action = env.action_space.sample()  # choose random action
        #action = policy_blue.gen_action(env.team1,observation,map_only=env.team_home)
        #action = [0, 0, 0, 0]
        #observation, reward, done, info = env.step(action)

        observation, reward, done, info = env.step()  # feedback from environment
        turnCounter += 1

        #will be storing observation and reward as a pattern set that the RL can learn from, essentially data colleciton
        individualTurns.append(observation.copy())
        individualTurnScores.append(reward)

        #limit to 25 turns to prevent stalling of the units
        if(turnCounter > turnLimit):
            done = True


        # render and sleep are not needed for score analysis
        env.render()
        time.sleep(.05)

        t += 1
        if t == 100000:
            break
    finalVal = len(individualTurnScores)
    
    while(len(individualTurnScores) <= turnLimit):
        individualTurnScores.append(individualTurnScores[finalVal - 1])
        individualTurns.append(individualTurns[finalVal - 1])
    
    overallScores.append(np.asanyarray(individualTurnScores))
    overallGames.append(np.asanyarray(individualTurns))

    rscore.pop(0)
    rscore.append(reward)
    env.reset()
    done = False
    print("Time: %.2f s, rscore: %.2f" %
        ((time.time() - start_time),sum(rscore)/len(rscore)))

print("check for exit")

#convert to numpy arrays to use in training models
overallGames = np.asanyarray(overallGames)
overallScores = np.asanyarray(overallScores)

#These results can be used to train the model next to make a new policy.
np.save('gameTrain.npy', overallGames)
np.save('gameResults.npy', overallScores)

print("Saved all the values/games")
