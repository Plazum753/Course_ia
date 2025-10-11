import pygame
from sys import exit
import math
import time
from info import plot
import random
import json
import copy

pygame.init()

vert=(0,180,0)
noir=(0,0,0)
blanc = (255,255,255)

Largeur = 976
Hauteur = 912

terrain = pygame.image.load('map.png') 
terrain = pygame.transform.scale(terrain, (Largeur, Hauteur))

# vitesse normal : 100
SPEED = 100

Zone_jeu = pygame.display.set_mode((Largeur,Hauteur))
pygame.display.set_caption("Course") #titre de la fenêtre


# def rejouerOUquitter():
#       for event in pygame.event.get([pygame.KEYDOWN,pygame.KEYUP,pygame.QUIT]): #si l'évènement est touche appuyé, relaché ou quit
#           if event.type == pygame.QUIT :
#               pygame.quit()
#               exit()
#           elif event.type == pygame.KEYUP : #touche relachée
#               continue
#           return 1 #event.key #on renvoie quelque chose différent de none
#       return None


# def gameOver():    
#     pygame.display.update() #Mise à jour de la fenêtre
    
#     while rejouerOUquitter()==None:
#         pass
#     car.reset() #Dès qu'on sort de la while, on relance le jeu
    
def affiche(text, position, color = noir, taille = 20):
    global affichage
    if affichage == True: 
        police=pygame.font.SysFont('arial',taille)
        texte=police.render(text,True,noir)
        zoneTexte = texte.get_rect()
        zoneTexte.center = position
        Zone_jeu.blit(texte,zoneTexte)
        #pygame.display.update() #Mise à jour de la fenêtre
    
class Car :
    def __init__(self,x=837,y=440,game_play = [], voiture = 'voiture.png', voiture_taille = 90):
        self.voitureLargeur=1920/voiture_taille #450/16
        self.voitureHauteur=1080/voiture_taille #204/16
        self.img=pygame.image.load(voiture) 
        self.img_origine = pygame.transform.scale(self.img, (self.voitureLargeur, self.voitureHauteur))
        
        self.x = x
        self.y = y
        
        self.game_play = game_play.copy()
        
        self.acceleration = 0.1
        self.frein = 0.4/3
        self.max_vitesse = 4
        self.frottement = 0.05/3
        self.maniabilité = 2
        self.adherence = 0.05
        
    def reset(self):
        self.img = self.img_origine 
        self.position = pygame.Vector2(self.x, self.y)  # Centre logique
        self.angle = 0
        self.tourner(90)
        self.game_over = False
        self.vitesse = 0         # Vitesse actuelle
        self.pos_old = pygame.Vector2(self.position[0],self.position[1])
        self.quart_tour = -1
        self.rotation_mouvement = 0
        self.tps_debut = -1
        self.n_frame = 0
        self.distance_parcouru = 0

        
        self.Jeux()
        
    def tourner(self, degre):
        self.angle += degre
        
        # Rotation de l'image
        self.img = pygame.transform.rotate(self.img_origine, self.angle)
        
        # Obtenir le nouveau rect et recentrer l'image
        self.rect = self.img.get_rect(center=self.position)
        
    def avancer(self):
        angle_rad = math.radians(-self.angle)
        
        direction = pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad))
        
        vitesse_actuelle = self.position - self.pos_old
        
        self.pos_old = pygame.Vector2(self.position[0],self.position[1])
        
        V_new = vitesse_actuelle + self.adherence*(direction * self.vitesse - vitesse_actuelle)
        
        if self.vitesse > 0:
            self.distance_parcouru += (V_new[0]**2+V_new[1]**2)**0.5
        
        self.position += V_new
        self.rect = self.img.get_rect(center=self.position)
        
    def collision(self, couleur):
        x, y = int(self.position.x), int(self.position.y)
        
        if not (0 <= x < Largeur and 0 <= y < Hauteur):
            return True
    
        pixel = terrain.get_at((x, y)) # donne la couleur du pixel du centre de la voiture
        
        count = 0
        for c in range(3):
            if couleur[c]-20<pixel[c]<couleur[c]+20:
                count+=1
        if count == 3:
            return True
        return False
    
    def tour(self):
        if self.quart_tour%4==3:
            if self.position[1]<425 and self.position[0]>2*Largeur/3: # en haut à droite
                self.quart_tour += 1
                
        if self.quart_tour%4==0:
            if self.position[1]<425 and self.position[0]<2*Largeur/3: # en haut à gauche
                self.quart_tour += 1      
                
        if self.quart_tour%4==1:
            if self.position[1]>425 and self.position[0]<2*Largeur/3: # en bas à gauche
                self.quart_tour += 1 
                    
        if self.quart_tour%4==2:
            if self.position[1]>425 and self.position[0]>2*Largeur/3: # en bas à droite
                self.quart_tour += 1
                
                
        if self.quart_tour%4==3:
            if self.position[1]>425 and self.position[0]<2*Largeur/3: # en bas à gauche
                self.quart_tour -= 1
                
        if self.quart_tour%4==0:
            if self.position[1]>425 and self.position[0]>2*Largeur/3: # en bas à droite
                self.quart_tour -= 1      
                
        if self.quart_tour%4==1:
            if self.position[1]<425 and self.position[0]>2*Largeur/3: # en haut à droite
                self.quart_tour -= 1 
                    
        if self.quart_tour%4==2:
            if self.position[1]<425 and self.position[0]<2*Largeur/3: # en haut à gauche
                self.quart_tour -= 1
    
    def fitness(self):
        self.perf = self.distance_parcouru/10**6
        self.perf += self.quart_tour
        if self.quart_tour>11:
            self.perf += 60
            self.perf -= self.n_frame/10**3
    
    def Jeux(self):
        if not self.game_over: #on demarre le jeu
            if len(self.game_play) == self.n_frame:
                self.game_play.append([random.choice([0,1]) for i in range(3)])
            
            if self.game_play[self.n_frame][0] == 1:
                avance = True
            else : 
                avance = False                          
            if self.game_play[self.n_frame][1] == 1:
                gauche = True
            else : 
                gauche = False
            if self.game_play[self.n_frame][2] == 1:
                droite = True
            else : 
                droite = False
            
            self.n_frame += 1


            if droite and not gauche:
                self.rotation_mouvement = -self.maniabilité
            elif gauche and not droite:
                self.rotation_mouvement = +self.maniabilité
            else:
                self.rotation_mouvement = 0
    
            # Accélération et freinage
            if avance:
                self.vitesse += self.acceleration
            else :
                self.vitesse -= self.frein
            
            # Limitation de la vitesse maximale
            if self.vitesse > self.max_vitesse:
                self.vitesse = self.max_vitesse
            if self.vitesse < 0:
                self.vitesse = 0  # Recul plus lent
                
            if abs(self.vitesse) > 0.2:
                #self.rotation_mouvement *= (self.vitesse / self.max_vitesse)
                self.tourner(self.rotation_mouvement)
                if self.tps_debut == -1 :
                    self.tps_debut = time.time()
                
            self.avancer()
                        
            if self.collision((12,190,0)):
                self.game_over = True
                self.fitness()

            self.tour()
            
            if self.quart_tour>11:
                self.tps = str(round(time.time()-self.tps_debut,1))
                self.game_over = True
                self.fitness()
                
                
def save(individu, filename="save.json"):
    try:
        with open(filename, "w") as f:
            json.dump(individu, f)
    except Exception as e:
        print("❌ Erreur de sauvegarde :", e)
        
def load(filename="save.json"):
    try:
        with open(filename, "r") as f:
            best_game_play = json.load(f)
        print("✅ Meilleure IA chargée depuis", filename)
        return best_game_play
    except Exception as e:
        print("❌ Erreur de chargement :", e)
        return None
                
                
def mutation(individu):
    global map_fini
    global fitness_plot
    
    new_genome = copy.deepcopy(individu)  #individu.copy()
    
    if len(new_genome) > 0 :
        if map_fini == True:
            for m in range(5):
                gene = random.randint(0, len(new_genome)-1)
                action = random.choice([0 for i in range(20)]+[1,2])
                # ancien_gene = new_genome[gene].copy()
                new_genome[gene][action] = abs(new_genome[gene][action]-1)
                # changement_vitesse = new_genome[gene][0]-ancien_gene[0]
                # if gene+2<len(new_genome):
                #     if changement_vitesse > 0:
                #         for _ in range(1):
                #             new_genome.pop(gene+1)
                #     elif changement_vitesse < 0:
                #         for _ in range(1):
                #             new_genome.insert(gene+1,new_genome[gene+1])
                
        elif len(new_genome) < 500:
            for m in range(1000):
                gene = random.randint(0, len(new_genome)-1)
                new_genome[gene] = [random.choice([0,1]) for _ in range(3)]

        else :
            changement = [random.choice([1,0])]+random.choice([[1,0],[0,1]])
            for m in range(1000):
                gene = random.randint(len(new_genome)-500, len(new_genome)-1)
                if random.randint(0,100) > 95:
                    new_genome[gene] = changement
                else :
                    new_genome[gene] = [random.choice([0,1]) for _ in range(3)]
      
    return new_genome

def cross_over(individu_1,individu_2):
    new_genome = []
    ancien_1 = individu_1.copy()
    ancien_2 = individu_2.copy()
    
    if len(ancien_1) < len(ancien_2):
        for gene in range(0,len(ancien_1),2):
            new_genome.append(ancien_1[gene])
            if gene + 1 < len(ancien_2):
                new_genome.append(ancien_2[gene+1])
        new_genome += ancien_2[len(ancien_1):]
    else : # >=
        for gene in range(0,len(ancien_2)-1,2):
            new_genome.append(ancien_2[gene])
            if gene + 1 < len(ancien_1):
                new_genome.append(ancien_1[gene+1])
        new_genome += ancien_1[len(ancien_2):]
    return new_genome


def new_pop():
    global pop
    global nom
    global stagnation
    global map_fini

    
    pop_copy = []
    for individu in range(len(pop)):
        pop_copy.append(pop[individu].game_play)
    pop = []
    pop.append(nom)
    pop[0] = Car(game_play= pop_copy[0], voiture = "f1.png", voiture_taille=60)
    nom += "a"
    

    for i in range(5*len(pop_copy)//6):
        pop.append(Car(game_play= mutation(cross_over(pop_copy[random.randint(0,len(pop_copy)//10)],pop_copy[random.randint(0,len(pop_copy)//10)]))))
    if stagnation > 3 and map_fini == False:
        stagnation = 0
        while len(pop)<len(pop_copy):
            if len(pop_copy[0])>2000:
                pop.append(Car(game_play= pop_copy[0][:-2000]))
    else :
        while len(pop)<len(pop_copy):
            pop.append(Car(game_play= mutation(pop_copy[random.randint(0,2)])))
    for i in range(len(pop)):
        pop[i].reset()
    
    
affichage = False #TODO
map_fini = False

# création de la première population
pop = []
nom = "a"
for i in range(300):
    pop.append(nom)
    nom += "a"
for individu in range(len(pop)):
    pop[individu] = Car()

sauvegarde = load()
if sauvegarde != None:
    pop[0]=Car(game_play=sauvegarde,voiture = "f1.png", voiture_taille=60)

for i in range(len(pop)):
    pop[i].reset()


fitness_moy_plot = []
fitness_plot = []
stagnation = 0

def train():
    global map_fini
    global stagnation
    
    print("génération n°"+str(len(fitness_plot)+1))

    fitness_tot = 0
    
    clock = pygame.time.Clock()
    fini = 0
    
    while fini < len(pop):
        for event in pygame.event.get():
            if event.type == pygame.QUIT : # lorsqu'on clique sur la croix rouge
                pygame.quit()
                exit()
        
        
        fini = 0
        Zone_jeu.blit(terrain,(0, 0)) 
        for i in range(len(pop)):
            if not pop[i].game_over:
                pop[i].Jeux()
                if affichage == True :
                    Zone_jeu.blit(pop[i].img,pop[i].rect.topleft) #on superpose la voiture dans la zone de jeu
            else:
                fini +=1
                
        if affichage == True :
            Zone_jeu.blit(pop[0].img,pop[0].rect.topleft) #on place la meilleur voiture au dessus des autres
                    
            if pop[0].quart_tour<0:
                affiche("Tour : 0/3",(Largeur-60,25))
            else:
                affiche("Tour :"+str(pop[0].quart_tour//4)+"/3",(Largeur-60,25))
            if pop[0].tps_debut == -1 :
                affiche("Chrono : 0s",(Largeur-75,50))
            else :
                affiche("Chrono :"+str(round(pop[0].n_frame/100,1))+"s",(Largeur-75,50))
            
            pygame.display.update() #on rafraichit l'écran.
            clock.tick(0)
    
    ancien_best = pop[0].game_play
    pop.sort(key=lambda v: v.perf,reverse=True)
    if ancien_best == pop[0].game_play:
        stagnation += 1
        
    if pop[0].quart_tour>11:
        map_fini = True
        nb_victoire = 0
        for individu in range(len(pop)):
            if pop[individu].quart_tour > 11:
                nb_victoire += 1
                fitness_tot += pop[individu].n_frame/100
        fitness_moy_plot.append(round(fitness_tot/nb_victoire,2))
        fitness_plot.append(round(pop[0].n_frame/100,2))
        plot(fitness_plot, fitness_moy_plot)
        print("temps : "+str(round(pop[0].n_frame/100,3)))
        print("taux = "+str(round(100*nb_victoire/len(pop),1))+"%")
    else :
        for individu in range(len(pop)):
            fitness_tot += pop[individu].distance_parcouru
        fitness_moy_plot.append(round(fitness_tot/len(pop),0))
        fitness_plot.append(round(pop[0].distance_parcouru,0))
        plot(fitness_plot, fitness_moy_plot)
        print("distance : "+str(round(pop[0].distance_parcouru,3)))
    save(pop[0].game_play)
    new_pop()
    
def test():
    global affichage
    affichage = True
    clock = pygame.time.Clock()

    pop[0].reset()
    while pop[0].game_over == False:
        for event in pygame.event.get():
            if event.type == pygame.QUIT : # lorsqu'on clique sur la croix rouge
                pygame.quit()
                exit()
        
        Zone_jeu.blit(terrain,(0, 0)) 
        pop[0].Jeux()
        Zone_jeu.blit(pop[0].img,pop[0].rect.topleft)
        if pop[0].quart_tour<0:
            affiche("Tour : 0/3",(Largeur-60,25))
        else:
            affiche("Tour :"+str(pop[0].quart_tour//4)+"/3",(Largeur-60,25))
        if pop[0].tps_debut == -1 :
            affiche("Chrono : 0s",(Largeur-75,50))
        else :
            affiche("Chrono :"+str(round(pop[0].n_frame/100,1))+"s",(Largeur-75,50))
        pygame.display.update()
        clock.tick(SPEED)
    affichage = False
#test()
while True:
    train()

pygame.quit() #on sort de la boucle donc on quitte
exit()