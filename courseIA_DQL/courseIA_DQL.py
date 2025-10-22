import pygame
from sys import exit
import math
import time
from info import plot
import random

import torch
import torch.optim as optim
import torch.nn as nn

from numba import njit
import numpy as np

pygame.init()

vert=(0,180,0)
noir=(0,0,0)
blanc = (255,255,255)

Largeur = 976
Hauteur = 912

terrain = pygame.image.load('Map.png') 
terrain = pygame.transform.scale(terrain, (Largeur, Hauteur))
terrain_array = pygame.surfarray.array3d(terrain).astype(np.int16)

# vitesse normal : 100
SPEED = 100

Zone_jeu = pygame.display.set_mode((Largeur,Hauteur))
pygame.display.set_caption("Course") #titre de la fenêtre

    
def affiche(text, position, color = noir, taille = 20):
    global affichage
    if affichage == True: 
        police=pygame.font.SysFont('arial',taille)
        texte=police.render(text,True,noir)
        zoneTexte = texte.get_rect()
        zoneTexte.center = position
        Zone_jeu.blit(texte,zoneTexte)
        #pygame.display.update() #Mise à jour de la fenêtre

# ============================= foction du programme ================================================

# distance
@njit
def dist(np_bord, np_bord_ext, new_dist, pos_x, pos_y, i_min, i_max, angle):
    dist_min = ((np_bord[new_dist][0]-pos_x)**2 + (np_bord[new_dist][1]-pos_y)**2)**0.5
    for i in range(i_min, i_max):
        if ((np_bord[i][0]-pos_x)**2 + (np_bord[i][1]-pos_y)**2)**0.5 < dist_min:
            dist_min = ((np_bord[i][0]-pos_x)**2 + (np_bord[i][1]-pos_y)**2)**0.5
            new_dist = i

    angle_normale = np.atan2(np_bord[new_dist][1]-np_bord_ext[new_dist][1],np_bord[new_dist][0]-np_bord_ext[new_dist][0])
    plus = (np_bord[new_dist][0]+10*np.cos(angle_normale+np.pi/2)-np_bord[new_dist+10][0])**2 + (np_bord[new_dist][0]+10*np.sin(angle_normale+np.pi/2)-np_bord[new_dist+10][1])**2 
    moins = (np_bord[new_dist][0]+10*np.cos(angle_normale-np.pi/2)-np_bord[new_dist+10][0])**2 + (np_bord[new_dist][0]+10*np.sin(angle_normale-np.pi/2)-np_bord[new_dist+10][1])**2 
    if plus > moins :
        angle_target = np.atan2(np_bord[new_dist][1]-np_bord_ext[new_dist][1],np_bord[new_dist][0]-np_bord_ext[new_dist][0])+np.pi/2
    else:
        angle_target = np.atan2(np_bord[new_dist][1]-np_bord_ext[new_dist][1],np_bord[new_dist][0]-np_bord_ext[new_dist][0])-np.pi/2
    angle_target = np.degrees(angle_target)

    diff_angle = angle_target - angle
    diff_angle = (diff_angle+540)%360-180 # intervalle [-180, 180[
    diff_angle = (diff_angle+180)/360 # normalisation  

    return new_dist, dist_min/100, diff_angle

# angle
@njit
def calcul_angle(np_bord, np_bord_ext, distance_parcouru):
    angle_normale = np.atan2(np_bord[distance_parcouru][1]-np_bord_ext[distance_parcouru][1],np_bord[distance_parcouru][0]-np_bord_ext[distance_parcouru][0])
    plus = (np_bord[distance_parcouru][0]+10*np.cos(angle_normale+np.pi/2)-np_bord[distance_parcouru+10][0])**2 + (np_bord[distance_parcouru][0]+10*np.sin(angle_normale+np.pi/2)-np_bord[distance_parcouru+10][1])**2 
    moins = (np_bord[distance_parcouru][0]+10*np.cos(angle_normale-np.pi/2)-np_bord[distance_parcouru+10][0])**2 + (np_bord[distance_parcouru][0]+10*np.sin(angle_normale-np.pi/2)-np_bord[distance_parcouru+10][1])**2 
    if plus > moins :
        angle_target = np.atan2(np_bord[distance_parcouru][1]-np_bord_ext[distance_parcouru][1],np_bord[distance_parcouru][0]-np_bord_ext[distance_parcouru][0])+np.pi/2
    else:
        angle_target = np.atan2(np_bord[distance_parcouru][1]-np_bord_ext[distance_parcouru][1],np_bord[distance_parcouru][0]-np_bord_ext[distance_parcouru][0])-np.pi/2
    angle_target = np.degrees(angle_target)
    
    return angle_target

# avancer
@njit
def calcul_avancer(pos, pos_old, angle, adherence, vitesse):
    angle_rad = math.radians(-angle)
            
    vitesse_actuelle = (pos[0] - pos_old[0], pos[1] - pos_old[1])
    
    V_new_x = vitesse_actuelle[0] + adherence*(math.cos(angle_rad) * vitesse - vitesse_actuelle[0])
    V_new_y= vitesse_actuelle[1] + adherence*(math.sin(angle_rad) * vitesse - vitesse_actuelle[1])
    
    new_pos = (pos[0] + V_new_x, pos[1] + V_new_y)
    
    return new_pos, (V_new_x**2 + V_new_y**2)**0.5

# tour
@njit
def tour(quart_tour,pos, Largeur):       
    if quart_tour%4==3:
        if pos[1]<425 and pos[0]>2*Largeur/3: # en haut à droite
            quart_tour += 1
            
    if quart_tour%4==0:
        if pos[1]<425 and pos[0]<2*Largeur/3: # en haut à gauche
            quart_tour += 1      
            
    if quart_tour%4==1:
        if pos[1]>425 and pos[0]<2*Largeur/3: # en bas à gauche
            quart_tour += 1 
                
    if quart_tour%4==2:
        if pos[1]>425 and pos[0]>2*Largeur/3: # en bas à droite
            quart_tour += 1
            
            
    if quart_tour%4==3:
        if pos[1]>425 and pos[0]<2*Largeur/3: # en bas à gauche
            quart_tour -= 1
            
    if quart_tour%4==0:
        if pos[1]>425 and pos[0]>2*Largeur/3: # en bas à droite
            quart_tour -= 1      
            
    if quart_tour%4==1:
        if pos[1]<425 and pos[0]>2*Largeur/3: # en haut à droite
            quart_tour -= 1 
                
    if quart_tour%4==2:
        if pos[1]<425 and pos[0]<2*Largeur/3: # en haut à gauche
            quart_tour -= 1
    return quart_tour
    
# =============================================================================


class Car :
    def __init__(self,x=837,y=440 ,Q_table = {}, voiture = 'f1.png', voiture_taille = 60):        
        self.voitureLargeur=1920/voiture_taille #450/16
        self.voitureHauteur=1080/voiture_taille #204/16
        self.img=pygame.image.load(voiture) 
        self.img_origine = pygame.transform.scale(self.img, (self.voitureLargeur, self.voitureHauteur))
        
        self.x = x
        self.y = y
        
        self.Q_table = Q_table.copy()

        self.acceleration = 0.1
        self.frein = 0.4/3
        self.max_vitesse = 4
        self.frottement = 0.05/3
        self.maniabilité = 2
        self.adherence = 0.05
        
        self.actions = ([1,0],[1,1],[1,2],[0,0],[0,1],[0,2])
        
        self.n_games = 0
        
        self.batch_size = 64
        
        self.replay_buffer = []
        self.replay_buffer_size = 50_000
        
        self.gamma = 0.99 # long ou court terme
                        
        self.model = nn.Sequential(
            nn.Linear(in_features=15,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=6)
            )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)  
        self.loss_criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
                                        
    def reset(self):            
        self.reward_tot = 0
        self.img = self.img_origine 
        
        self.angle = 0
        self.position = pygame.Vector2(self.x, self.y)  # Centre logique
        self.tourner(90)

        self.game_over = False
        self.vitesse = 0         # Vitesse actuelle
        self.pos_old = self.position
        self.quart_tour = -1
        self.rotation_mouvement = 0
        self.tps_debut = -1
        self.n_frame = 0
        self.n_games += 1
        self.reward = 0
        self.pause = 0
        self.dist_bord = 2.0
        self.dist_centre_old = 10
        self.diff_angle = 0
        self.diff_angle_old = 0
        self.distance_parcouru = 0
        self.distance()
        self.reward = 0
        self.n_mort = 0
                
        self.Jeux()
    
    def distance(self):
        global bord
        global np_bord
        global np_bord_ext
        
        new_dist = self.distance_parcouru

        i_min = max(0, self.distance_parcouru - 100)
        i_max = min(len(bord), self.distance_parcouru + 200)
        
        new_dist, self.dist_bord, self.diff_angle = dist(np_bord, np_bord_ext, new_dist, self.position.x, self.position.y, i_min, i_max, self.angle)
        
        
        self.reward += (new_dist-self.distance_parcouru)*0.001
        
        self.distance_parcouru = new_dist

            
    def tourner(self, degre):
        global affichage
        self.angle += degre
        self.angle = self.angle%360
        
        # Rotation de l'image
        if affichage == True:
            self.img = pygame.transform.rotate(self.img_origine, self.angle)
        
            # Obtenir le nouveau rect et recentrer l'image
            self.rect = self.img.get_rect(center=self.position)
        
    def avancer(self):
        new_position, V = calcul_avancer((self.position.x,self.position.y), (self.pos_old.x,self.pos_old.y), self.angle, self.adherence, self.vitesse)
        self.pos_old = self.position
        self.position = pygame.Vector2(new_position[0],new_position[1])
        if affichage == True:
            self.rect = self.img.get_rect(center=self.position)
        
        if V < 0.1:
            self.pause += 1
            if self.pause > 100 :
                self.reward -= 1
        else : 
            self.pause = 0
        
    def collision(self, couleur=(12,190,0), x=None, y=None): 
        if x == None:
            x, y = int(self.position.x), int(self.position.y)
        if not (0 <= x < Largeur and 0 <= y < Hauteur):
            return True    
        return np.all(np.abs(terrain_array[x, y] - couleur) < 25)
            
    def Jeux(self):
        global map_fini
        global np_bord_ext
        global np_bord
        
        if not self.game_over: 
            
# =============================== récupération de state ================================================

            vitesse_actuelle = self.position - self.pos_old
                     
            state = [(vitesse_actuelle.x**2+vitesse_actuelle.y**2)**0.5/4,
                     self.dist_bord,
                     self.diff_angle]
            for i in range(0, 120, 40):
                state += [np_bord[self.distance_parcouru+i][0], 
                          np_bord[self.distance_parcouru+i][1],
                          np_bord_ext[self.distance_parcouru+i][0],
                          np_bord_ext[self.distance_parcouru+i][1]]
                
# =============================== récupération de l'action =============================================
            
            epsilon = max(2,100-self.n_games)#TODO
            
            if random.randint(0,100) > epsilon :
                with torch.no_grad():
                    action = torch.argmax(self.model(torch.tensor(state))).item()
            else :
                action = random.randint(0,5) 

            if self.actions[action][0] == 1:
                avance = True
            else : 
                avance = False                          
            if self.actions[action][1] == 1:
                gauche = True
            else : 
                gauche = False
            if self.actions[action][1] == 2:
                droite = True
            else : 
                droite = False
                
# =============================== éxécution de l'action ================================================
            self.reward = -0.001
                
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
            if self.vitesse < 0 :
                self.vitesse = 0  
                
            if abs(self.vitesse) > 0.2:
                #self.rotation_mouvement *= (self.vitesse / self.max_vitesse)
                self.tourner(self.rotation_mouvement)
                if self.tps_debut == -1 :
                    self.tps_debut = time.time()
                
            self.avancer()
            
            self.distance()
            
            if self.collision() :
                #self.game_over = True
                self.n_mort += 1
                self.reward -= 1
                self.distance_parcouru += 10
                
                x = (np_bord[self.distance_parcouru][0]+np_bord_ext[self.distance_parcouru][0])/2
                y = (np_bord[self.distance_parcouru][1]+np_bord_ext[self.distance_parcouru][1])/2  
                
                self.position = pygame.Vector2(x, y)
                self.pos_old = pygame.Vector2(x, y)
                pygame.draw.rect(terrain, (255,0,0), (np_bord[self.distance_parcouru][0],np_bord[self.distance_parcouru][1],2,2), 2)
                pygame.draw.rect(terrain, (255,0,0), (np_bord_ext[self.distance_parcouru][0],np_bord_ext[self.distance_parcouru][1],2,2), 2)
                self.angle = calcul_angle(np_bord, np_bord_ext, self.distance_parcouru)
                self.vitesse = 0

            self.quart_tour = tour(self.quart_tour,(self.position.x, self.position.y), Largeur)

            if self.quart_tour>11:
                self.tps = str(round(time.time()-self.tps_debut,1))
                self.game_over = True
            #print(state,self.actions[action],self.reward)
            
# =============================== mise à jour du model ============================================

            vitesse_actuelle = self.position - self.pos_old
                     
            state_new = [(vitesse_actuelle.x**2+vitesse_actuelle.y**2)**0.5/4,
                     self.dist_bord,
                     self.diff_angle]
            for i in range(0, 120, 40):
                state_new += [np_bord[self.distance_parcouru+i][0], 
                          np_bord[self.distance_parcouru+i][1],
                          np_bord_ext[self.distance_parcouru+i][0],
                          np_bord_ext[self.distance_parcouru+i][1]]

            self.replay_buffer.append((state, action, self.reward, state_new))
            if len(self.replay_buffer) > self.replay_buffer_size :
                self.replay_buffer.pop(0)
                
            if self.n_frame % 4 == 0 and len(self.replay_buffer) > self.batch_size :
                batch = random.sample(self.replay_buffer, self.batch_size)
                
                states = torch.tensor([e[0] for e in batch], dtype=torch.float32).to(self.device)
                actions = torch.tensor([e[1] for e in batch], dtype=torch.long).unsqueeze(1).to(self.device)
                rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32).to(self.device)
                next_states = torch.tensor([e[3] for e in batch], dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    esperances_new_state = self.model(next_states)
                    max_esperances, max_actions = esperances_new_state.max(dim=1)
                    target = rewards + self.gamma * max_esperances 
                    
                esperances = self.model(states)
                pred = esperances.gather(1, actions).squeeze(1) 
                
                loss = self.loss_criterion(pred, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            self.reward_tot += self.reward
            self.n_frame += 1

def isbord(x, y, couleur=(95,207,43)):   
    return np.all(np.abs(terrain_array[x, y] - couleur) < 50)
    
def point_new(point_old):
    test = ((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1))    

    compteur = 0
    while not isbord(point_old[0]+test[compteur][0], point_old[1]+test[compteur][1]) :
        compteur = (compteur+1)%8
    while isbord(point_old[0]+test[compteur][0], point_old[1]+test[compteur][1]) :
        compteur = (compteur+1)%8
    return (point_old[0]+test[compteur][0], point_old[1]+test[compteur][1])

def centre_piste():
    bord = []

    point = (812,412) #­ premier point de collision à gauche de la ligne d'arrivée

    while point not in bord :
        bord.append(point)
        point = point_new(point)
        #pygame.draw.rect(terrain, (255,0,0), (point[0],point[1],2,2), 2)
    bord_ext0 = []
    point = (884,412)
    while point not in bord_ext0 :
        bord_ext0.append(point)
        point = point_new(point)
        #pygame.draw.rect(terrain, (255,0,0), (point[0],point[1],2,2), 2)
    
    ind = len(bord_ext0)-1
    bord_ext = []
    for p in bord : 
        mini = bord_ext0[ind-1]
        if (mini[0]-p[0])**2+(mini[1]-p[1])**2 < (bord_ext0[ind][0]-p[0])**2+(bord_ext0[ind][1]-p[1])**2 :
            ind -= 1
        ind -= 1
        bord_ext.append(bord_ext0[ind+1])
    bord2 = tuple(bord[-100:]+bord+bord+bord+bord[:200]) # il faut une marge | bord[-100:] la voiture commence avant la ligne d'arrivée
    bord_ext2 = tuple(bord_ext[-100:]+bord_ext+bord_ext+bord_ext+bord_ext[:200]) 
    print(len(bord2),len(bord_ext2))
    return bord2, bord_ext2

bord, bord_ext = centre_piste()
np_bord = np.array(bord)
np_bord_ext = np.array(bord_ext)

def save(car, filename="save.pth"):
    torch.save(car.model.state_dict(), filename)
        
def load(car, filename="save.pth"):
    try:
        car.model.load_state_dict(torch.load(filename, map_location=car.device))
        car.model.to(car.device)
        print(f"✅ model chargée depuis {filename}")
    except Exception as e:
        print("❌ Erreur lors du chargement du model :", e)
        return None
    
    
affichage = True #TODO
map_fini = False

car = Car()

sauvegarde = load(car, filename="save.pth")
    

scores_moy_plot = []
scores_plot = []
score_moy = []
reward_moy_plot = []
reward_moy = []
record = 10**6

def train():
    global map_fini
    global record
    global affichage
    
    ancien_affichage = affichage
    
    print("Game n°"+str(len(scores_plot)+1))

    clock = pygame.time.Clock()
    car.reset()
    if car.n_games%10==0:
        affichage = True
        save(car, filename="save.pth")
        
    while not car.game_over :
        for event in pygame.event.get():
            if event.type == pygame.QUIT : # lorsqu'on clique sur la croix rouge
                pygame.quit()
                exit()
                
        Zone_jeu.blit(terrain,(0, 0)) 
        car.Jeux()
        
        if affichage == True :
                
            Zone_jeu.blit(car.img,car.rect.topleft) #on place la meilleur voiture au dessus des autres
                    
            if car.quart_tour<0:
                affiche("Tour : 0/3",(Largeur-60,25))
            elif car.quart_tour > 11:
                affiche("Tour : 3/3",(Largeur-60,25))
            else:
                affiche("Tour :"+str(car.quart_tour//4+1)+"/3",(Largeur-60,25))
            if car.tps_debut == -1 :
                affiche("Chrono : 0s",(Largeur-75,50))
            else :
                affiche("Chrono :"+str(round(car.n_frame/100,1))+"s",(Largeur-75,50))
            
            pygame.display.update() #on rafraichit l'écran.
            clock.tick(0)
        
    scores_plot.append(round(car.n_frame/100,2))
    print("temps : "+str(round(car.n_frame/100,2)))
    if car.n_frame/100 < record and car.n_mort == 0:
        record = car.n_frame/100
    print("record : "+str(round(record,2))+"s")
    print()
    reward_moy.append(car.n_mort) #car.reward_tot
    score_moy.append(scores_plot[-1])
    while len(score_moy)>100:
        score_moy.pop(0)
    while len(reward_moy)>100:
        reward_moy.pop(0)
    scores_moy_plot.append(round(sum(score_moy)/len(score_moy),2))
    reward_moy_plot.append(round(sum(reward_moy)/len(reward_moy),2))
    
    plot(scores_plot,scores_moy_plot,reward_moy_plot)
    
    if ancien_affichage != affichage:
        affichage = ancien_affichage
        
def test():
    global affichage
    ancien_affichage = affichage
    affichage = True
    clock = pygame.time.Clock()

    car.reset()
    while car.game_over == False:
        for event in pygame.event.get():
            if event.type == pygame.QUIT : # lorsqu'on clique sur la croix rouge
                pygame.quit()
                exit()
        
        Zone_jeu.blit(terrain,(0, 0)) 
        car.Jeux()
        Zone_jeu.blit(car.img,car.rect.topleft)
        if car.quart_tour<0:
            affiche("Tour : 0/3",(Largeur-60,25))
        else:
            affiche("Tour :"+str(car.quart_tour//4)+"/3",(Largeur-60,25))
        if car.tps_debut == -1 :
            affiche("Chrono : 0s",(Largeur-75,50))
        else :
            affiche("Chrono :"+str(round(car.n_frame/100,1))+"s",(Largeur-75,50))
        pygame.display.update()
        clock.tick(SPEED)
        
    if car.quart_tour>11:
        print("temps : "+str(round(car.n_frame/100,2)))
    else :
        print("distance : "+str(round(car.distance_parcouru,2)))
        
    affichage = ancien_affichage
    
#test()

while True:
    train()

pygame.quit() #on sort de la boucle donc on quitte
exit()