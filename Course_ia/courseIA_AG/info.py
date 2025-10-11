import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(fitness, fitness_moy):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Générations')
    plt.ylabel('Temps')
    plt.plot(fitness,"r*")
    plt.plot(fitness_moy,"b*")
    plt.ylim(ymin=0)
    plt.text(len(fitness)-1,fitness[-1],str(fitness[-1]))
    plt.text(len(fitness_moy)-1, fitness_moy[-1], str(fitness_moy[-1]))