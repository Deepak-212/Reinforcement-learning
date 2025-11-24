import numpy as np
# numberofBins - this is a 4 dimensional list that defines the number of grid points for state discretization

class Q_learning:

    def  __init__(self,env,alpha,gamma,epsilon,numberEpisodes,numberofBins,lowerBounds,upperBounds):
        import numpy as np

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actionNumber = env.action_space.n
        self.numberEpisodes = numberEpisodes
        self.numberofBins = numberofBins
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds

        self.sumRewardsEpisode = []

        self.Qmatrix = np.random.uniform(low=0, high=1, size=(numberofBins[0],numberofBins[1],numberofBins[2],numberofBins[3],self.actionNumber))

    def returnIndexState(self,state):
        #position =        state[0]
        #velocity =        state[1]
        #angle =           state[2]
        #angularvelocity = state[3]

        cartPositionBin = np.linspace(self.lowerBounds[0],self.upperBounds[0],self.numberofBins[0])
        cartVelocityBin = np.linspace(self.lowerBounds[1],self.upperBounds[1],self.numberofBins[1])
        poleAngleBin = np.linspace(self.lowerBounds[2],self.upperBounds[2],self.numberofBins[2])
        poleAngularVelocityBin = np.linspace(self.lowerBounds[3],self.upperBounds[3],self.numberofBins[3])

        indexPosition = np.maximum(np.digitize(state[0],cartPositionBin)-1,0)
        indexVelocity = np.maximum(np.digitize(state[1],cartVelocityBin)-1,0)
        indexAngle = np.maximum(np.digitize(state[2],poleAngleBin)-1,0)
        indexAngularVelocity = np.maximum(np.digitize(state[3],poleAngularVelocityBin)-1,0)

        return tuple([indexPosition,indexVelocity,indexAngle,indexAngularVelocity])
    
    def selectAction(self,state,index):

        if index<500:
            return np.random.choice(self.actionNumber)
        
        randomNumber = np.random.random()
        if index>7000:
            self.epsilon=0.999*self.epsilon
        if randomNumber < self.epsilon:
            return np.random.choice(self.actionNumber)
        else:
            return np.random.choice(np.where(self.Qmatrix[self.returnIndexState(state)]==np.max(self.Qmatrix[self.returnIndexState(state)]))[0])
    
    def simulateEpisodes(self):
        import numpy as np

        for indexEpisode in range(self.numberEpisodes):

            rewardsEpisode=[]
            (stateS,_)=self.env.reset()
            stateS=list(stateS)
            print("Simulating episode {}".format(indexEpisode))

            terminalState=False
            while not terminalState:
                stateSIndex=self.returnIndexState(stateS)
                actionA = self.selectAction(stateS,indexEpisode)
                (stateSprime, reward, terminalState,_,_) = self.env.step(actionA)
                rewardsEpisode.append(reward)
                stateSprime=list(stateSprime)
                stateSprimeIndex=self.returnIndexState(stateSprime)
                QmaxPrime=np.max(self.Qmatrix[stateSprimeIndex])                                               
                                              
                if not terminalState:
                    # stateS+(actionA,) - we use this notation to append the tuples
                    # for example, for stateS=(0,0,0,1) and actionA=(1,0)
                    # we have stateS+(actionA,)=(0,0,0,1,0)
                    error=reward+self.gamma*QmaxPrime-self.Qmatrix[stateSIndex+(actionA,)]
                    self.Qmatrix[stateSIndex+(actionA,)]=self.Qmatrix[stateSIndex+(actionA,)]+self.alpha*error
                else:
                    # in the terminal state, we have Qmatrix[stateSprime,actionAprime]=0 
                    error=reward-self.Qmatrix[stateSIndex+(actionA,)]
                    self.Qmatrix[stateSIndex+(actionA,)]=self.Qmatrix[stateSIndex+(actionA,)]+self.alpha*error
                 
                # set the current state to the next state                    
                stateS=stateSprime
                print(f'Sum of rewards {np.sum(rewardsEpisode)}')
                self.sumRewardsEpisode.append(np.sum(rewardsEpisode))

    def simulateLearnedStrategy(self):
        import gymnasium as gym
        import time
        env1 = gym.make("CartPole-v1", render_mode="human")
        (currentState,_)=env1.reset()
        timeSteps=1000
        # obtained rewards at every time step
        obtainedRewards=[]

        for timeIndex in range(timeSteps):
            print(timeIndex)
            # select greedy actions
            actionInStateS=np.random.choice(np.where(self.Qmatrix[self.returnIndexState(currentState)]==np.max(self.Qmatrix[self.returnIndexState(currentState)]))[0])
            currentState, reward, terminated, truncated, info =env1.step(actionInStateS)
            obtainedRewards.append(reward)   
            time.sleep(0.05)
            if (terminated):
                time.sleep(1)
                break
        return obtainedRewards,env1



             




