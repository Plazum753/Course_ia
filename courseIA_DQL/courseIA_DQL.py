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

import cProfile, pstats

profiler = cProfile.Profile()
profiler.enable()

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

# ============================= foction du programme ================================================

# distance
@njit
def dist(np_bord, np_bord_ext, new_dist, pos_x, pos_y, i_min, i_max, angle, angles):
    dist_min = ((np_bord[new_dist][0]-pos_x)**2 + (np_bord[new_dist][1]-pos_y)**2)**0.5
    for i in range(i_min, i_max):
        if ((np_bord[i][0]-pos_x)**2 + (np_bord[i][1]-pos_y)**2)**0.5 < dist_min:
            dist_min = ((np_bord[i][0]-pos_x)**2 + (np_bord[i][1]-pos_y)**2)**0.5
            new_dist = i

    diff_angle = angles[new_dist] - angle
    diff_angle = (diff_angle+540)%360-180 # intervalle [-180, 180[
    diff_angle = (diff_angle+180)/360 # normalisation  

    return new_dist, dist_min/100, diff_angle

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
        
        self.gamma = 0.99 # long ou court terme
                        
        self.model = nn.Sequential(
            nn.Linear(in_features=15,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=6)
            )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0050)  
        self.loss_criterion = nn.MSELoss()
        self.device = torch.device("cpu")      #"cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.batch_size = 256
        
        self.buffer_max = 50_000
        self.buffer_size = 0
        self.buffer_index = 0
        
        self.buffer_state = np.zeros((self.buffer_max, 15))
        self.buffer_action = np.zeros((self.buffer_max,))
        self.buffer_reward = np.zeros((self.buffer_max,))
        self.buffer_state_new = np.zeros((self.buffer_max, 15))
        
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
        global angles
        
        i_min = max(0, self.distance_parcouru - 10)
        i_max = min(len(bord), self.distance_parcouru + 10)
        
        new_dist, self.dist_bord, self.diff_angle = dist(np_bord, np_bord_ext, int(self.distance_parcouru), np.float32(self.position.x), np.float32(self.position.y), int(i_min), int(i_max), int(self.angle), angles)
        
        
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
        new_position, V = calcul_avancer((np.float32(self.position.x),np.float32(self.position.y)), (np.float32(self.pos_old.x),np.float32(self.pos_old.y)), int(self.angle), np.float32(self.adherence), np.float32(self.vitesse))
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
        global  np_bord 
        global np_bord_ext 
        global angles
        
        if not self.game_over: 
            
# =============================== récupération de state ================================================

            vitesse_actuelle = self.position - self.pos_old
                     
            state = [float((vitesse_actuelle.x**2+vitesse_actuelle.y**2)**0.5/4),
                     float(self.dist_bord),
                     float(self.diff_angle)]
            for i in range(0, 120, 40):
                state += [float(np_bord[self.distance_parcouru+i][0]/Largeur), 
                          float(np_bord[self.distance_parcouru+i][1]/Hauteur),
                          float(np_bord_ext[self.distance_parcouru+i][0]/Largeur),
                          float(np_bord_ext[self.distance_parcouru+i][1]/Hauteur)]
                
# =============================== récupération de l'action =============================================
            
            epsilon = max(1,100-self.n_games)
            
            if random.randint(0,100) > epsilon :
                with torch.inference_mode():
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
                
                #self.distance_parcouru -= 50
                self.distance_parcouru = max(0, self.distance_parcouru)
                                
                x = (np_bord[self.distance_parcouru][0]+np_bord_ext[self.distance_parcouru][0])/2
                y = (np_bord[self.distance_parcouru][1]+np_bord_ext[self.distance_parcouru][1])/2  
                
                self.position = pygame.Vector2(x, y)
                self.pos_old = pygame.Vector2(x, y)

                # pygame.draw.rect(terrain, (255,0,0), (np_bord[self.distance_parcouru][0],np_bord[self.distance_parcouru][1],2,2), 2)
                # pygame.draw.rect(terrain, (255,0,0), (np_bord_ext[self.distance_parcouru][0],np_bord_ext[self.distance_parcouru][1],2,2), 2)
                self.angle = angles[self.distance_parcouru]

                self.vitesse = 0

            self.quart_tour = tour(int(self.quart_tour),(np.float32(self.position.x), np.float32(self.position.y)), Largeur)

            if self.quart_tour>11:
                self.tps = str(round(time.time()-self.tps_debut,1))
                self.game_over = True
            #print(state,self.actions[action],self.reward)
            
# =============================== mise à jour du model ============================================

            vitesse_actuelle = self.position - self.pos_old
                     
            state_new = [float((vitesse_actuelle.x**2+vitesse_actuelle.y**2)**0.5/4),
                     float(self.dist_bord),
                     float(self.diff_angle)]
            for i in range(0, 120, 40):
                state_new += [float(np_bord[self.distance_parcouru+i][0]/Largeur), 
                              float(np_bord[self.distance_parcouru+i][1]/Hauteur),
                              float(np_bord_ext[self.distance_parcouru+i][0]/Largeur),
                              float(np_bord_ext[self.distance_parcouru+i][1]/Hauteur)]
            
            self.buffer_state[self.buffer_index] = state
            self.buffer_action[self.buffer_index] = action
            self.buffer_reward[self.buffer_index] = self.reward
            self.buffer_state_new[self.buffer_index] = state_new
                
            self.buffer_index = (self.buffer_index + 1) % self.buffer_max
            self.buffer_size = min(self.buffer_size+1,self.buffer_max)
            
            if self.n_frame % 16 == 0 and self.buffer_size > self.batch_size :
                indices = np.random.randint(0, self.buffer_size, size=self.batch_size)
            
                states = torch.as_tensor(self.buffer_state[indices], dtype=torch.float32, device=self.device)
                actions = torch.as_tensor(self.buffer_action[indices], dtype=torch.long, device=self.device)
                rewards = torch.as_tensor(self.buffer_reward[indices], dtype=torch.float32, device=self.device)
                next_states = torch.as_tensor(self.buffer_state_new[indices], dtype=torch.float32, device=self.device)
                
                with torch.inference_mode():
                    esperances_new_state = self.model(next_states)
                    max_esperances, max_actions = esperances_new_state.max(dim=1)
                    target = rewards + self.gamma * max_esperances 
                target = target.clone().detach()
                    
                esperances = self.model(states)
                pred = esperances.gather(1, actions.unsqueeze(1)).squeeze(1) 
                
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

@njit
def meme_taille(A,B):
    compteur = 0
    C = []
    if len(A) > len(B):
        frequence = len(A)/(len(A)-len(B))
        for i in range(len(A)):
            if i > compteur :
                compteur += frequence
            else :
                C.append(A[i])
        return C,B
    elif len(A) < len(B):
        frequence = len(B)/(len(B)-len(A))
        for i in range(len(B)):
            if i > compteur :
                compteur += frequence
            else :
                C.append(B[i])
        return A,C
    return A,B
    

def centre_piste():
    bord0 = []

    point = (812,412) #­ premier point de collision à gauche de la ligne d'arrivée

    while point not in bord0 :
        bord0.append(point)
        point = point_new(point)
        #pygame.draw.rect(terrain, (255,0,0), (point[0],point[1],2,2), 2)
    bord_ext0 = []
    point = (884,412)
    while point not in bord_ext0 :
        bord_ext0.append(point)
        point = point_new(point)
        #pygame.draw.rect(terrain, (255,0,0), (point[0],point[1],2,2), 2)
    bord_ext0.reverse()    
    
    indices = ((0,0),(100,100),(170,310),(300,460),(740,850),(800,1100),(850,1250),(1100,1525),(1250,1610),(1415,1660),(1550,1810),(1605,2000),(1700,2125),(2070,2500),(2190,2700),(2250,2860),(2340,2915),(-1,-1))
    
    # for i in indices :
    #     pygame.draw.rect(terrain, (255,0,0), (bord0[i[0]][0],bord0[i[0]][1],4,4), 2)
    #     pygame.draw.rect(terrain, (255,0,0), (bord_ext0[i[1]][0],bord_ext0[i[1]][1],4,4), 2)
    
    bord = []   
    bord_ext = []
    for i in range(len(indices)-1):
        b,b_ext = meme_taille(bord0[indices[i][0]:indices[i+1][0]], bord_ext0[indices[i][1]:indices[i+1][1]])
        bord += b
        bord_ext += b_ext
       
    # for point in bord:
    #     pygame.draw.rect(terrain, (255,0,0), (point[0],point[1],2,2), 2)
    # for point in bord_ext:
    #     pygame.draw.rect(terrain, (255,0,0), (point[0],point[1],2,2), 2)
    
    bord = tuple(bord[-100:]+bord+bord+bord+bord[:200]) # il faut une marge | bord[-100:] la voiture commence avant la ligne d'arrivée
    bord_ext = tuple(bord_ext[-100:]+bord_ext+bord_ext+bord_ext+bord_ext[:200]) 
    return bord, bord_ext

bord, bord_ext = centre_piste()
np_bord = np.array(bord, dtype=np.float32)
np_bord_ext = np.array(bord_ext, dtype=np.float32)

@njit
def liste_angles(np_bord, np_bord_ext):
    angles = []
    for i in range(len(np_bord)) : 
        cx1 = (np_bord[i][0]+np_bord_ext[i][0])/2
        cy1 = (np_bord[i][1]+np_bord_ext[i][1])/2
        cx2 = (np_bord[i+1][0]+np_bord_ext[i+1][0])/2
        cy2 = (np_bord[i+1][1]+np_bord_ext[i+1][1])/2
        dx = cx2 - cx1
        dy = cy2 - cy1
        angle_target = np.degrees(np.atan2(-dy, dx))%360
        angles.append(angle_target)
    return np.array(angles, dtype = np.float32)

angles = liste_angles(np_bord, np_bord_ext)

def save(car, filename="save.pth"):
    torch.save({
    'model': car.model.state_dict(),
    'optimizer': car.optimizer.state_dict(),
    'n_game': car.n_games}, filename)
        
def load(car, filename="save.pth"):
    try:
        sauvegarde = torch.load(filename)
        car.model.load_state_dict(sauvegarde["model"])
        car.model.to(car.device)
        car.optimizer.load_state_dict(sauvegarde["optimizer"])
        #car.optimizer = optim.Adam(car.model.parameters(), lr=0.0005) # pour changer le learning rate
        car.n_games = sauvegarde["n_game"]
        print(f"✅ model chargée depuis {filename}")
    except Exception as e:
        print("❌ Erreur lors du chargement du model :", e)
        return None
    
    
affichage = False #TODO
map_fini = False

car = Car()

sauvegarde = load(car, filename="save.pth")
#car.model = torch.compile(car.model)
    

scores_moy_plot = []
reward_plot = []
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
    

    #clock = pygame.time.Clock()
    car.reset()
    print("Game n°"+str(car.n_games))

    if car.n_games%20==0:
        affichage = True
        save(car, filename="save.pth")
    
    while not car.game_over :
        if car.n_frame % 100 == 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT : # lorsqu'on clique sur la croix rouge
                    pygame.quit()
                    exit()
                    
        if affichage == True :
            Zone_jeu.blit(terrain,(0, 0)) 
        car.Jeux()
        
        if affichage == True :
                
            Zone_jeu.blit(car.img,car.rect.topleft) #on place la meilleur voiture au dessus des autres
            # if car.quart_tour<0:
            #     affiche("Tour : 0/3",(Largeur-60,25))
            # elif car.quart_tour > 11:
            #     affiche("Tour : 3/3",(Largeur-60,25))
            # else:
            #     affiche("Tour :"+str(car.quart_tour//4+1)+"/3",(Largeur-60,25))
            # if car.tps_debut == -1 :
            #     affiche("Chrono : 0s",(Largeur-75,50))
            # else :
            #     affiche("Chrono :"+str(round(car.n_frame/100,1))+"s",(Largeur-75,50))
            
            pygame.display.update() #on rafraichit l'écran.
            #clock.tick(0)
        
    scores_plot.append(round(car.n_frame/100,2))
    print("temps : "+str(round(car.n_frame/100,2)))
    if car.n_frame/100 < record and car.n_mort == 0:
        record = car.n_frame/100
    print("record : "+str(round(record,2))+"s")
    print()
    reward_moy.append(car.n_mort) #car.reward_tot
    reward_plot.append(car.n_mort)
    score_moy.append(scores_plot[-1])
    while len(score_moy)>100:
        score_moy.pop(0)
    while len(reward_moy)>100:
        reward_moy.pop(0)
    scores_moy_plot.append(round(sum(score_moy)/len(score_moy),2))
    reward_moy_plot.append(round(sum(reward_moy)/len(reward_moy),2))
    
    plot(scores_plot,scores_moy_plot,reward_plot, reward_moy_plot)
    
    if ancien_affichage != affichage:
        affichage = ancien_affichage
        
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(16)  # Affiche les 50 fonctions les plus lentes
    
while True:
    train()

pygame.quit() #on sort de la boucle donc on quitte
exit()