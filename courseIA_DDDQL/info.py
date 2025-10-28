import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(score, score_moy, reward, reward_moy):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of games')
    plt.ylabel('Temps')
    plt.plot(score,"g*", label="temps en secondes")    
    plt.plot(score_moy,"g", label="temps moyen en secondes")
    plt.plot(reward,"r*", label="nombre d'accidents")
    plt.plot(reward_moy,"r", label="nombre moyen d'accidents")
    plt.ylim(ymin=0,ymax=100)
    plt.xlim(left=0)
    if score[-1] < 100 :
        plt.text(len(score)-1,score[-1],str(score[-1]))
    if reward[-1] < 100 :
        plt.text(len(reward)-1,reward[-1],str(reward[-1]))
    if reward_moy[-1] < 100 :
        plt.text(len(reward_moy)-1,reward_moy[-1],str(reward_moy[-1]))
    if score_moy[-1] < 100 :
        plt.text(len(score_moy)-1, score_moy[-1], str(score_moy[-1]))
    plt.legend(loc="upper left")
