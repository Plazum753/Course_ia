import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(score, score_moy, reward):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of games')
    plt.ylabel('Temps')
    plt.plot(score,"g*")
    plt.plot(reward,"r", label="nombre d'accidents")
    plt.plot(score_moy,"g", label="temps en secondes")
    plt.ylim(ymin=0,ymax=100)
    plt.xlim(left=0)
    if score[-1] < 100 :
        plt.text(len(score)-1,score[-1],str(score[-1]))
    if reward[-1] < 100 :
        plt.text(len(reward)-1,reward[-1],str(reward[-1]))
    if score_moy[-1] < 100 :
        plt.text(len(score_moy)-1, score_moy[-1], str(score_moy[-1]))
    plt.legend(loc="upper left")
