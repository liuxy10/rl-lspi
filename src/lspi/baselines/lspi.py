from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

Sample = namedtuple('Sample', ['s', 'a', 'r', 's_', 'done'])


class LSPolicyIteration:
    def __init__(self,
                 env,
                 agent,
                 gamma,
                 memory_size,
                 memory_type='sample',
                 eval_type='batch'):
        """Least-Squares Policy Iteration algorithm

        Args:
            env (gym.Env): gym environment.
            agent (lspi.agents.Agent): features policy.
            gamma (float): discount factor.
            memory_size (int): number of training samples/episodes.
            memory_type (str, optional): samples collecting method. Defaults to 'sample'.
            eval_type (str, optional): policy evaluation method. Defaults to 'batch'.
        """
        if not memory_type in ['sample', 'episode']:
            raise ValueError(
                "memory_type can take values ['sample','episode']")
        if not eval_type in ['iterative', 'sherman_morrison', 'batch']:
            raise ValueError(
                "eval_type can take values ['iterative','sherman_morrison','batch']"
            )
        self.env = env
        self.gamma = gamma
        self.agent = agent
        self.memory_size = memory_size
        self.eval_type = eval_type
        self.memory_type = memory_type

    def init_memory(self, agent = None, update_size = 10):
        """Initialize the memory with random samples (if policy is None) or update memory or episodes."""
        if agent is None:
            self.memory = []
        obs = self.env.reset()
        count = 0 # count of collected samples
        done, done_idx = True, []
        limit = (self.memory_size + 1) if agent is None else update_size
        while count < limit:
            if done:
                done_idx.append(len(self.memory))
                obs = self.env.reset()
                if self.memory_type == 'episode':
                    count += 1
                    # print("percentage done = {:.2f}%".format(count / (self.memory_size + 1) * 100)) if agent is None else print("epi update done = {:.2f}%".format(count / update_size * 100))
            action = self.env.action_space.sample() if agent is None else agent.predict(obs)
            next_obs, reward, done, _ = self.env.step(action)
            self.memory.append(Sample(obs, action, reward, next_obs, done))
            obs = next_obs
            if self.memory_type == 'sample':
                count += 1
                # print("percentage done = {:.2f}%".format(count / (self.memory_size + 1) * 100)) if agent is None else print("sample update done = {:.2f}%".format(count / update_size * 100))
        # only keep latest memory size
        # print("memory size before trimming = {}".format(len(self.memory)), "done index", done_idx)
        self.memory = self.memory[-self.memory_size:] if self.memory_type == 'sample' else self.memory[ -done_idx[0]:]

        if self.eval_type == 'batch':
            self._batch_prep()

    def _batch_prep(self):
        if self.agent.__class__.__name__ == "QuadraticAgent":
                # raise NotImplementedError("Batch method not implemented for Quadratic Agent yet")
            k = self.agent.features_size
            self.A_all = np.zeros((len(self.memory), k, k))
            self.b_all = np.zeros(k)
            for idx, sample in enumerate(self.memory):
                    # state features
                feat_s = self.agent.get_features(sample.s, sample.a)
                    # next state features
                feat_s_ = self.agent.get_features(sample.s_, self.agent.predict(sample.s_))
                self.A_all[idx, :, :] = np.outer(
                        feat_s, feat_s - self.gamma * feat_s_)
                    # reward features
                self.b_all += sample.r * feat_s
            self.b_all = self.b_all.reshape(-1,1)
        else:
            k = self.agent.features_size
            nActions = self.agent.action_size
            self.A_all = np.zeros(
                    (len(self.memory), nActions, k * nActions, k * nActions))
            self.b_all = np.zeros(k * nActions)
            for idx, sample in enumerate(self.memory):
                    # state features
                feat_s = np.zeros(k * nActions)
                a = sample.a
                feat_s[a * k:(a + 1) * k] = self.agent.get_features(sample.s,sample.a)
                    # next state features
                feat_ = self.agent.get_features(sample.s_, self.agent.predict(sample.s_))
                for a_ in range(nActions):
                    feat_s_ = np.zeros(k * nActions)
                    feat_s_[a_ * k:(a_ + 1) * k] = feat_
                    self.A_all[idx, a_, :, :] = np.outer(
                            feat_s, feat_s - self.gamma * feat_s_)
                    # reward features
                self.b_all += sample.r * feat_s

    def load_memory(self, memory, vis = False):
        self.memory = memory
        if vis:
            # print state (normalized), reward trajectory in 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            visited_states, rewards, dones = [], [], []
            for sample in memory:
                visited_states.append(sample.s)
                rewards.append(sample.r)
                dones.append(sample.done)
            ax1.plot(visited_states)
            ax1.set_title("Visited States Over Time")
            ax1.set_xlabel("Time Step")
            ax1.set_ylabel("State Value")
            ax2.plot(rewards)
            ax2.hlines(0, 0, len(rewards), colors='r', linestyles='dashed')
            ax2.set_title("Rewards Over Time")
            ax2.set_xlabel("Time Step")


            ax2.set_ylabel("Reward Value")
            plt.tight_layout()
            plt.show()

            
        

 

    def eval(self):
        k = self.agent.features_size
        nActions = self.agent.action_size
        if self.eval_type == 'iterative':
            if self.agent.__class__.__name__ == "QuadraticAgent":
                A = np.zeros((k,k))
                b = np.zeros((k,1))
                for sample in tqdm(self.memory, desc="LSTDQ"):
                    # state features
                    feat_s = self.agent.get_features(sample.s, sample.a).reshape(-1,1)
                    # next state features
                    a_ = self.agent.predict(sample.s_)
                    feat_s_ = self.agent.get_features(sample.s_, a_).reshape(-1,1)
                    # update parameters
                    A += np.outer(feat_s, feat_s - self.gamma * feat_s_)
                    b += sample.r * feat_s

                

            else:
                A = np.zeros((k * nActions, k * nActions))
                b = np.zeros(k * nActions)
                for sample in self.memory:
                    # state features
                    feat_s = np.zeros(k * nActions)
                    a = sample.a
                    feat_s[a * k:(a + 1) * k] = self.agent.get_features(sample.s,a)
                    # next state features
                    feat_s_ = np.zeros(k * nActions)
                    a_ = self.agent.predict(sample.s_)
                    feat_s_[a_ * k:(a_ + 1) * k] = self.agent.get_features(
                        sample.s_, a_)
                    # update parameters
                    A += np.outer(feat_s, feat_s - self.gamma * feat_s_)
                    b += sample.r * feat_s
                    # w = np.linalg.solve(A, b)
        elif self.eval_type == 'sherman_morrison':
            if self.agent.__class__.__name__ == "QuadraticAgent":
                raise NotImplementedError("Sherman-Morrison not implemented for Quadratic Agent yet")
                
            else: 
                B = np.eye(k * nActions)
                b = np.zeros(k * nActions)
                for sample in self.memory:
                    # state features
                    feat_s = np.zeros(k * nActions)
                    a = sample.a
                    feat_s[a * k:(a + 1) * k] = self.agent.get_features(sample.s, a)
                    # next state features
                    feat_s_ = np.zeros(k * nActions)
                    a_ = self.agent.predict(sample.s_)
                    feat_s_[a_ * k:(a_ + 1) * k] = self.agent.get_features(
                        sample.s_, a_)
                    # update matrix
                    B -= np.outer(np.dot(
                        B, feat_s), np.dot(
                            B.T, feat_s - self.gamma * feat_s_)) / (1 + np.inner(
                                feat_s - self.gamma * feat_s_, np.dot(B, feat_s)))
                    b += sample.r * feat_s
                w = np.dot(B, b)
        elif self.eval_type == 'batch':
            if self.agent.__class__.__name__ == "QuadraticAgent":
                # raise NotImplementedError("Batch method not implemented for Quadratic Agent yet")
                A = self.A_all.sum(0)
                b = self.b_all
                w = np.linalg.solve(A, b)
            else:
                A = np.array([
                    self.A_all[idx, self.agent.predict(sample.s_)]
                    for idx, sample in enumerate(self.memory)
                ]).sum(0)
                b = self.b_all
                # w = np.linalg.solve(A, b)
        
        
        ap = True # use alternative projection
        if not ap:
            w = np.linalg.solve(A, b).reshape(-1) if np.linalg.det(A) != 0 else np.linalg.pinv(A).dot(b).reshape(-1)
            # make sure the solution is positive definite by setting negative eigenvalues to negative
            S = -self.agent.convertW2S(w)
            D, V = np.linalg.eigh(S)
            D = np.maximum(D, 1e-3)
            S = V @ np.diag(D) @ V.T
            w = -self.agent.convertS2W(S)
        else:
            j=1
            error=1
            stepsize=0.5 * 0.5
            nStates = len(self.memory[0].s)
            phiw=np.zeros((k,1))
            Cphi=A/nStates
            dphi=b/nStates
            while j<=1000 and error>1e-4:
                oPhiw=phiw.copy()
                residuePhi=phiw-stepsize*(Cphi@phiw-dphi) 
                phiw=proDysktra(self.agent, residuePhi,100,1e-4)
                j=j+1
                error=np.linalg.norm(oPhiw-phiw)
                stepsize=1/(j+1)
            w=phiw.reshape(-1) 
        
        
        return w

    def train_step(self):
        w = self.eval()
        # print("Updated policy weights:", w)
        self.agent.set_weights(w)

# alternating projection
def proDysktra(agent,x0,ballR,errTol):
    """Projection onto the set of symmetric matrices."""    

    error=1
    j=1
    I= np.zeros((len(x0),2)) # intermediate projections
    oldI=np.zeros((len(x0),2)) 
    x=x0.reshape((-1,))
    while j<500 and error>errTol: 
        
        oldX=x.copy()
        if np.linalg.norm(x-I[:,0])>ballR:
            x=ballR*(x-I[:,0])/np.linalg.norm(x-I[:,0])                    
        else:
            x=x-I[:,0]               
        oldI[:,0]=I[:,0].copy()
        I[:,0]=x-(oldX-I[:,0])
        
        oldX=x.copy()
        s=-agent.convertW2S(x-I[:,1])
        D, V = np.linalg.eigh(s)  # D is diagonal matrix, V is orthogonal
        D[D< 0]=0    # set negative eigenvalues to zero                          
        s=V@np.diag(D)@V.T
        x=-agent.convertS2W(s) # x is the new point
        oldI[:,1]=I[:,1].copy()
        I[:,1]=x-(oldX-I[:,1])  
                        
        j=j+1
        error=np.linalg.norm(oldI-I)**2                
            
    return x.reshape(-1,1)  # return the projection of x0 onto the set of symmetric matrices

        
