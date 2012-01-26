import numpy
import random

"""Library to implement hidden Markov Models"""
class Probability(object):
    """Represents a probability as a callable object"""
    
    def __init__(self,n,d):
        """Initialises the probability from a numerator and a denominator"""
        self.Numerator=float(n)
        self.Denominator=float(d)

    def __call__(self):
        """Returns the value of the probability"""
        return self.Numerator/self.Denominator

    def Update(self,deltaN,deltaD):
        """Updates the probability during Bayesian learning"""
        self.Numerator+=deltaN
        self.Denominator+=deltaD

    def __iadd__(self,Prob2):
        """Updates the probability given another Probability object"""
        self.Numerator+=Prob2.Numerator
        self.Denominator+=Prob2.Denominator
        return self

class Distribution(object):
    """ Represents a probability distribution over a set of categories"""
    def __init__(self,categories,k=0):
        """The distribution may be initialised from a list of categories or a
           dictionary of category frequencies. In the latter case, Laplacian
           smoothing may be used"""
        self.Probs={}
        if type(categories).__name__=='dict':
            Denominator=float(len(categories)*k)
            for item in categories:
                Denominator+=float(categories[item])
            self.Probs=dict([(item,Probability(categories[item]+k,Denominator)) for item in categories])
        else:
            Denominator=len(categories)
            self.Probs=dict([(item,Probability(1,Denominator)) for item in categories])

    def __call__(self,item):
        """Gives the probability of item"""
        return self.Probs[item]()

    def __mult__(self,scalar):
        """Returns the probability of each item, multiplied by a scalar"""
        return dict([(item,self.Probs[item]()*scalar) for item in self.Probs])

    def Update(self,categories):
        """Updates each category in the probability distiribution, according to
           a dictionary of numerator and denominator values"""
        for item in categories:
            self.Probs[item].Update(categories[item])

    def Sample(self):
        """Picks a random sample from the distribution"""
        p=random.random()
        Seeking=True
        result=None
        for item in self.Probs:
            x=self(item)
            if x>p:
                Seeling=False
                result=item
            else:
                p-=x
        return result

    def States(self):
        return [state for state in self.Probs]

class PoissonDistribution(Distribution):
    """Represents a Poisson distribution"""
    def __init__(self,mean):
        """Initialises the distribution with a given mean"""
        self.Numerator=float(mean)
        self.Denominator=1.0

    def __call__(self,N):
        """Returns the probability of N"""
        logProb=numpy.ln(self.Numerator)-numpy.ln(self.Denominator)
        logProb*=N
        logProb-=(self.Numerator/self.Denominator)
        for i in range(2,N+1):
            logProb-=numpy.ln(i)
        return numpy.exp(logProb)

    def Update(self,N,p=1.0):
        """Updates the distribution, given a value N that has a probability of P
           of being drawn from this distribution"""
        self.Numerator+=N*p
        self.Denominator+=p

    def Mean(self):
        return self.Numerator/self.Denominator

    def Sample(self):
        """Returns a random sample from the Poisson distribution"""
        p=random.random()
        n=0
        Seeking=True
        while Seeking:
            x=self(n)
            if x>p:
                Seeking=False
            else:
                n+=1
                p-=x
        return n

class BayesianModel(object):
    """Represents a Bayesian probability model"""
    def __init__(self,Prior,Conditionals):
        """Prior is a Distribution. Conditionals is a dictionary mapping
           each state in Prior to a Distribution"""
        self.Prior=Prior
        self.Conditionals=Conditionals

    def __call__(self,PriorProbs=None):
        """Returns a Distribution representing the probabilities of the outcomes
           given a particular distribution of the priors, which defaults to
           self.Prior"""
        if PriorProbs==None:
            PriorProbs=self.Prior
        Outcomes={}
        for state in PriorProbs:
            posterior=self.Conditionals[state]*self.Priors(state)
            for outcome in posterior:
                Outcomes.setdefault(outcome,0.0)
                Outcomes[outcome]+=posterior[outcome]
        return Distribution(outcomes)

    def PriorProbs(self,Observations):
        """Returns a Distribution representing the probabilities of the prior
           states, given a probability Distribution of Observations"""
        return Distribution((((state,self.Priors(state)*sum((self.Conditionals[state][outcome]()*Observations[outcome]() for outcome in Observations)) for state in self.Priors))))
        
        

class HMM(BayesianModel):
    """Represents a Hidden Markov Model"""
    def __init__(self,states,outcomes):
        """states is a list or dictionary of states, outcomes is a dictionary
           mapping each state in states to a distribution of the output states"""
        super(HMM,self).__init__(Distribution(states,1),outcomes)
        states=self.Prior.States()
        self.TransitionProbs=BayesianModel(states,dict([(state,Distribution(states)) for state in states]))
        
           
        
    
            
