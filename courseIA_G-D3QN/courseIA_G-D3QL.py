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

# =============================================================================
# import cProfile, pstats
# 
# profiler = cProfile.Profile()
# profiler.enable()
# =============================================================================


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
        texte=police.render(text,True,color)
        zoneTexte = texte.get_rect()
        zoneTexte.center = position
        Zone_jeu.blit(texte,zoneTexte)
        
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
    'model_feature1': car.model1_features.state_dict(),
    'model_valeur1': car.model1_valeur.state_dict(),
    'model_avantages1': car.model1_avantages.state_dict(),
    
    'model_feature2': car.model2_features.state_dict(),
    'model_valeur2': car.model2_valeur.state_dict(),
    'model_avantages2': car.model2_avantages.state_dict(),
    
    'optimizer1': car.optimizer1.state_dict(),
    'optimizer2': car.optimizer2.state_dict(),
    
    'n_game': car.n_games}, filename)
    print(f"💾 Modèle sauvegardé dans {filename}")
    
def load(car, filename="save.pth"):
    try:
        sauvegarde = torch.load(filename)
        car.model1_features.load_state_dict(sauvegarde["model_feature1"])
        car.model1_valeur.load_state_dict(sauvegarde["model_valeur1"])
        car.model1_avantages.load_state_dict(sauvegarde["model_avantages1"])
        
        car.model2_features.load_state_dict(sauvegarde["model_feature2"])
        car.model2_valeur.load_state_dict(sauvegarde["model_valeur2"])
        car.model2_avantages.load_state_dict(sauvegarde["model_avantages2"])
        
        car.model1_features.to(car.device)
        car.model1_valeur.to(car.device)
        car.model1_avantages.to(car.device)
        car.model2_features.to(car.device)
        car.model2_valeur.to(car.device)
        car.model2_avantages.to(car.device)
        
        car.optimizer1.load_state_dict(sauvegarde["optimizer1"])
        car.optimizer2.load_state_dict(sauvegarde["optimizer2"])
        
        car.n_games = sauvegarde["n_game"]
        
        print(f"✅ model chargée depuis {filename}")
    except Exception as e:
        print("❌ Erreur lors du chargement du model :", e)
        return None


# ============================= foction du programme ================================================

# distance
@njit
def dist(np_bord, np_bord_ext, new_dist, pos_x, pos_y, i_min, i_max):
    dist_min = ((np_bord[new_dist][0]-pos_x)**2 + (np_bord[new_dist][1]-pos_y)**2)**0.5
    for i in range(i_min, i_max):
        if ((np_bord[i][0]-pos_x)**2 + (np_bord[i][1]-pos_y)**2)**0.5 < dist_min:
            dist_min = ((np_bord[i][0]-pos_x)**2 + (np_bord[i][1]-pos_y)**2)**0.5
            new_dist = i

    return new_dist

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

# state
@njit
def point_state(pos, angle_voiture, point):    
    norme_normalise = ((point[0] - pos[0])**2 + (point[1] - pos[1])**2)**0.5/100
    
    angle_rad = np.radians(-angle_voiture)
    angle_point = np.arctan2(point[1] - pos[1], point[0] - pos[0])
    angle = angle_point - angle_rad
    angle_normalise = ((angle + 3*np.pi) % (2*np.pi)) / (2*np.pi)
    # (|entre -pi et pi| (angle + 3*pi) % (2*pi) - pi |normalisation| + pi) / (pi -(-pi))
    
    return [np.float32(angle_normalise), np.float32(norme_normalise)]

@njit
def vecteur_vitesse(vx,vy, angle_voiture):
    norme_normalise = (vx**2 + vy**2)**0.5/4
    angle_rad = np.radians(-angle_voiture)
    direction = np.arctan2(vy,vx)
    angle = direction - angle_rad
    angle_normalise = ((angle + 3*np.pi) % (2*np.pi)) / (2*np.pi)
    
    return [np.float32(angle_normalise), np.float32(norme_normalise)]
    
    
# =============================================================================

class Car :
    def __init__(self,x=837,y=440, voiture = 'voiture.png', voiture_taille = 90):        
        self.voitureLargeur=1920/voiture_taille #450/16
        self.voitureHauteur=1080/voiture_taille #204/16
        self.img=pygame.image.load(voiture) 
        self.img_origine = pygame.transform.scale(self.img, (self.voitureLargeur, self.voitureHauteur))
        
        self.x = x
        self.y = y
        
        self.acceleration = 0.1
        self.frein = 0.4/3
        self.max_vitesse = 4
        self.frottement = 0.05/3
        self.maniabilité = 2
        self.adherence = 0.05
        
        self.actions = ([1,0],[1,1],[1,2],[0,0],[0,1],[0,2])
        
        self.n_games = 0
        
        self.gamma = 0.99 # long ou court terme

# =============================================================================
        self.model1_features = nn.Sequential( 
            nn.Linear(in_features=14,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=256),
            nn.ReLU(),
            )
        self.model1_valeur = nn.Sequential(
            nn.Linear(in_features=256,out_features=256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1)
            )
        self.model1_avantages = nn.Sequential(
            nn.Linear(in_features=256,out_features=256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=6)
            )
# =============================================================================
        self.model2_features = nn.Sequential(
            nn.Linear(in_features=14,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=256),
            nn.ReLU(),
            )
        self.model2_valeur = nn.Sequential(
            nn.Linear(in_features=256,out_features=256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1)
            )
        self.model2_avantages = nn.Sequential(
            nn.Linear(in_features=256,out_features=256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=6)
            )
# =============================================================================
        self.learning_rate = 10**-5
        self.optimizer1 = optim.Adam(
            list(self.model1_features.parameters()) +
            list(self.model1_valeur.parameters()) +
            list(self.model1_avantages.parameters()),
            lr=self.learning_rate
            )  
        self.optimizer2 = optim.Adam(
            list(self.model2_features.parameters()) +
            list(self.model2_valeur.parameters()) +
            list(self.model2_avantages.parameters()),
            lr=self.learning_rate
            )  
        
        self.loss_criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1_features.to(self.device)
        self.model1_valeur.to(self.device)
        self.model1_avantages.to(self.device)
        self.model2_features.to(self.device)
        self.model2_valeur.to(self.device)
        self.model2_avantages.to(self.device)

        
        self.batch_size = 256
        
        self.buffer_max = 50_000
        self.buffer_size = 0
        self.buffer_index = 0
        
        self.buffer_state = np.zeros((self.buffer_max, 14))
        self.buffer_action = np.zeros((self.buffer_max,))
        self.buffer_reward = np.zeros((self.buffer_max,))
        self.buffer_state_new = np.zeros((self.buffer_max, 14))
        
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
        self.distance_parcouru = 0
        self.distance()
        self.reward = 0
        self.n_mort = 0
                
        self.Jeux()
    
    def distance(self):
        global np_bord
        global np_bord_ext
        global angles
        
        i_min = max(0, self.distance_parcouru - 10)
        i_max = min(len(np_bord), self.distance_parcouru + 10)
        
        new_dist = dist(np_bord, np_bord_ext, int(self.distance_parcouru), np.float32(self.position.x), np.float32(self.position.y), int(i_min), int(i_max))
        
        self.reward += (new_dist-self.distance_parcouru)*0.005
        
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
        global  np_bord 
        global np_bord_ext 
        global angles
        
        if not self.game_over: 
            
# =============================== récupération de state ================================================

            vitesse_actuelle = self.position - self.pos_old
                     
            state = vecteur_vitesse(np.float32(vitesse_actuelle[0]), np.float32(vitesse_actuelle[1]), int(self.angle))
            for i in range(0, 120, 40):
                state += point_state((np.float32(self.position.x),np.float32(self.position.y)), int(self.angle), np_bord[self.distance_parcouru+i])
                state += point_state((np.float32(self.position.x),np.float32(self.position.y)), int(self.angle), np_bord_ext[self.distance_parcouru+i])

# =============================== récupération de l'action =============================================
            
            epsilon = max(2,100-self.n_games)
            
            if random.randint(0,100) > epsilon :
                with torch.inference_mode():
                    features1 = self.model1_features(torch.tensor(state, dtype=torch.float32, device=self.device))
                    avantages1 = self.model1_avantages(features1)
                    features2 = self.model2_features(torch.tensor(state, dtype=torch.float32, device=self.device))
                    avantages2 = self.model2_avantages(features2)
                    action = torch.argmax((avantages1+avantages2)/2).item()
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
            self.reward = -0.005
                
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
                     
            state_new = vecteur_vitesse(np.float32(vitesse_actuelle[0]), np.float32(vitesse_actuelle[1]), int(self.angle))
            for i in range(0, 120, 40):
                state_new += point_state((np.float32(self.position.x),np.float32(self.position.y)), int(self.angle), np_bord[self.distance_parcouru+i])
                state_new += point_state((np.float32(self.position.x),np.float32(self.position.y)), int(self.angle), np_bord_ext[self.distance_parcouru+i])

            
            self.buffer_state[self.buffer_index] = state
            self.buffer_action[self.buffer_index] = action
            self.buffer_reward[self.buffer_index] = self.reward
            self.buffer_state_new[self.buffer_index] = state_new
                
            self.buffer_index = (self.buffer_index + 1) % self.buffer_max
            self.buffer_size = min(self.buffer_size+1,self.buffer_max)
            
            if self.n_frame % 128 == 0 and self.buffer_size > self.batch_size : #TODO essayer de repasser à 16
                indices = np.random.randint(0, self.buffer_size, size=self.batch_size)
            
                states = torch.as_tensor(self.buffer_state[indices], dtype=torch.float32, device=self.device)
                actions = torch.as_tensor(self.buffer_action[indices], dtype=torch.long, device=self.device)
                rewards = torch.as_tensor(self.buffer_reward[indices], dtype=torch.float32, device=self.device)
                next_states = torch.as_tensor(self.buffer_state_new[indices], dtype=torch.float32, device=self.device)
                
                if random.randint(0,1) == 0:
                    # mise à jour du model 1
                    with torch.inference_mode():
                        # choix de la prochaine action
                        next_features_choix = self.model1_features(next_states)
                        next_action = self.model1_avantages(next_features_choix)
                        _, max_next_action = next_action.max(dim=1)
                        
                        # calcul de target
                        next_features = self.model2_features(next_states)
                        next_valeur = self.model2_valeur(next_features).squeeze(1)
                        next_avantages = self.model2_avantages(next_features)
                        
                        next_avantage = next_avantages.gather(1, max_next_action.unsqueeze(1)).squeeze(1)
                        mean_next_avantages = next_avantages.mean(dim=1)
                        target = rewards + self.gamma * (next_valeur + next_avantage - mean_next_avantages)
                    target = target.clone().detach()
                    
                    features = self.model1_features(states)
                    valeur = self.model1_valeur(features).squeeze(1)
                    avantages = self.model1_avantages(features)
                    mean_avantages = avantages.mean(dim=1)
                    avantage = avantages.gather(1, actions.unsqueeze(1)).squeeze(1)
                    
                    pred = valeur + avantage - mean_avantages
                    
                    loss = self.loss_criterion(pred, target)
                    self.optimizer1.zero_grad()
                    loss.backward()
                    self.optimizer1.step()
                else :
                    # mise à jour du model 2
                    with torch.inference_mode():
                        # choix de la prochaine action
                        next_features_choix = self.model2_features(next_states)
                        next_action = self.model2_avantages(next_features_choix)
                        _, max_next_action = next_action.max(dim=1)
                        
                        # calcul de target
                        next_features = self.model1_features(next_states)
                        next_valeur = self.model1_valeur(next_features).squeeze(1)
                        next_avantages = self.model1_avantages(next_features)
                        
                        next_avantage = next_avantages.gather(1, max_next_action.unsqueeze(1)).squeeze(1)
                        mean_next_avantages = next_avantages.mean(dim=1)
                        target = rewards + self.gamma * (next_valeur + next_avantage - mean_next_avantages)
                    target = target.clone().detach()
                    
                    features = self.model2_features(states)
                    valeur = self.model2_valeur(features).squeeze(1)
                    avantages = self.model2_avantages(features)
                    mean_avantages = avantages.mean(dim=1)
                    avantage = avantages.gather(1, actions.unsqueeze(1)).squeeze(1)
                    
                    pred = valeur + avantage - mean_avantages
                    
                    loss = self.loss_criterion(pred, target)
                    self.optimizer2.zero_grad()
                    loss.backward()
                    self.optimizer2.step()
                
            self.reward_tot += self.reward
            self.n_frame += 1
    

def crea_model():
        model1_features = nn.Sequential( 
            nn.Linear(in_features=14,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=256),
            nn.ReLU(),
            )
        model1_valeur = nn.Sequential(
            nn.Linear(in_features=256,out_features=256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1)
            )
        model1_avantages = nn.Sequential(
            nn.Linear(in_features=256,out_features=256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=6)
            )
# =============================================================================
        model2_features = nn.Sequential(
            nn.Linear(in_features=14,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=256),
            nn.ReLU(),
            )
        model2_valeur = nn.Sequential(
            nn.Linear(in_features=256,out_features=256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1)
            )
        model2_avantages = nn.Sequential(
            nn.Linear(in_features=256,out_features=256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=6)
            )
        
        model = [
            model1_features,
            model1_valeur,
            model1_avantages,
            model2_features,
            model2_valeur,
            model2_avantages
            ]
        return model

def mutation(A):
    for model in [A.model1_features, A.model1_valeur, A.model1_avantages, A.model2_features, A.model2_valeur, A.model2_avantages] :
        for param in model.parameters():
            if random.random() > 0.9 :
                param.data *= random.choice([0.75,1.25])
    if random.random() > 0.5 :
        A.learning_rate *= random.choice([0.5,2])
        
        A.optimizer1 = optim.Adam(
            list(A.model1_features.parameters()) +
            list(A.model1_valeur.parameters()) +
            list(A.model1_avantages.parameters()),
            lr=A.learning_rate
            )  
        A.optimizer2 = optim.Adam(
            list(A.model2_features.parameters()) +
            list(A.model2_valeur.parameters()) +
            list(A.model2_avantages.parameters()),
            lr=A.learning_rate
            )  
    
    return A
                
def cross_over(L):
    A = Car()
    model = crea_model()
    
    model0 = [
        L[0].model1_features,
        L[0].model1_valeur,
        L[0].model1_avantages,
        L[0].model2_features,
        L[0].model2_valeur,
        L[0].model2_avantages
        ]
    model1 = [
        L[1].model1_features,
        L[1].model1_valeur,
        L[1].model1_avantages,
        L[1].model2_features,
        L[1].model2_valeur,
        L[1].model2_avantages
        ]
    
    for m in range(6) :
        for param, param0, param1 in zip(model[m].parameters(), model0[m].parameters(), model1[m].parameters()):
            param.data = random.choice([param0.data,param1.data])
    
    A.learning_rate = random.choice([L[0].learning_rate, L[1].learning_rate])
    
    A.model1_features = model[0]
    A.model1_valeur = model[1]
    A.model1_avantages = model[2]
    A.model2_features = model[3]
    A.model2_valeur = model[4]
    A.model2_avantages = model[5]
    
    A.model1_features.to(A.device)
    A.model1_valeur.to(A.device)
    A.model1_avantages.to(A.device)
    A.model2_features.to(A.device)
    A.model2_valeur.to(A.device)
    A.model2_avantages.to(A.device)
    
    A.optimizer1 = optim.Adam(
        list(A.model1_features.parameters()) +
        list(A.model1_valeur.parameters()) +
        list(A.model1_avantages.parameters()),
        lr=A.learning_rate
        )
    A.optimizer2 = optim.Adam(
        list(A.model2_features.parameters()) +
        list(A.model2_valeur.parameters()) +
        list(A.model2_avantages.parameters()),
        lr=A.learning_rate
        )    
    A.n_games = L[0].n_games
    
    return A
    
def new_population(population):
    new_population = []
    
    for i in range(3) :
        population[i].voitureLargeur=1920/60
        population[i].voitureHauteur=1080/60
        population[i].img=pygame.image.load("f1.png") 
        population[i].img_origine = pygame.transform.scale(population[0].img, (population[0].voitureLargeur, population[0].voitureHauteur))

        new_population.append(population[i])
        
    while len(new_population) != len(population) :
        new_population.append(mutation(cross_over(random.sample(population[:6],2))))
    

    
    return new_population

# création de la première population
population = []
for i in range(20):
    population.append(Car())

for i in range(len(population)):
    load(population[i])

affichage = False

scores_moy_plot = []
reward_plot = []
scores_plot = []
score_moy = []
reward_moy_plot = []
reward_moy = []
record = 10**6

def train():
    global record
    global affichage
    global population
        
    print("Generation n°"+str(population[0].n_games//20))
    #clock = pygame.time.Clock()
    
    for i in range(20):
        print("Game n°"+str(population[0].n_games))

        for i in range(len(population)):
            population[i].reset()
    
        if population[0].n_games%20==0:
            save(population[0], filename="save.pth")
            
# =============================================================================
#         profiler = cProfile.Profile()
#         profiler.enable()
# =============================================================================
        
        fini = 0
        while fini < len(population) :
            for event in pygame.event.get():
                if event.type == pygame.QUIT : # lorsqu'on clique sur la croix rouge
                    pygame.quit()
                    exit()
            
            if population[-1].n_frame%100 == 0:
                affichage = True
                        
            if affichage == True :
                Zone_jeu.blit(terrain,(0, 0)) 
            for i in range(len(population)):
                if not population[i].game_over:
                    population[i].Jeux()
                    if affichage == True :
                        Zone_jeu.blit(population[i].img,population[i].rect.topleft) #on superpose la voiture dans la zone de jeu
                else:
                    fini +=1  
                    
            if affichage == True :
                    
                Zone_jeu.blit(population[0].img,population[0].rect.topleft) #on place la meilleur voiture au dessus des autres
                if population[0].quart_tour<0:
                    affiche("Tour : 0/3",(Largeur-60,25))
                elif population[0].quart_tour > 11:
                    affiche("Tour : 3/3",(Largeur-60,25))
                else:
                    affiche("Tour :"+str(population[0].quart_tour//4+1)+"/3",(Largeur-60,25))
                if population[0].tps_debut == -1 :
                    affiche("Chrono : 0s",(Largeur-75,50))
                else :
                    affiche("Chrono :"+str(round(population[0].n_frame/100,1))+"s",(Largeur-75,50))
                
                pygame.display.update() #on rafraichit l'écran.
                #clock.tick(0)
                
# =============================================================================
#             if population[0].n_frame%1000==0:
#                 profiler.disable()
#                 stats = pstats.Stats(profiler).sort_stats('cumtime')
#                 stats.print_stats(50)  # Affiche les 50 fonctions les plus lentes
# =============================================================================
                
            affichage = False
            
        population.sort(key=lambda v: v.reward_tot,reverse=True)
        
        if population[0].n_frame/100 < record and population[0].n_mort == 0:
            record = population[0].n_frame/100
        
    print("temps : "+str(round(population[0].n_frame/100,2)))
    print("record : "+str(round(record,2))+"s")
    print()
    scores_plot.append(round(population[0].n_frame/100,2))
    reward_moy.append(round(sum([population[i].n_mort for i in range(len(population))])/len(population),2)) #car.reward_tot
    reward_plot.append(population[0].n_mort)
    score_moy.append(round(sum([population[i].n_frame/100 for i in range(len(population))])/len(population),2))

    scores_moy_plot.append(score_moy)
    reward_moy_plot.append(reward_moy)
    
    plot(scores_plot,scores_moy_plot,reward_plot, reward_moy_plot)
    
    population = new_population(population)
    
# =============================================================================
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats(16)  # Affiche les 50 fonctions les plus lentes
# =============================================================================

def test():
    global affichage
    affichage = True
    clock = pygame.time.Clock()

    population[0].reset()
    while not population[0].game_over :
        for event in pygame.event.get():
            if event.type == pygame.QUIT :
                pygame.quit()
                exit()
        
        Zone_jeu.blit(terrain,(0, 0)) 
        population[0].Jeux()
        Zone_jeu.blit(population[0].img,population[0].rect.topleft)
        if population[0].quart_tour<0:
            affiche("Tour : 0/3",(Largeur-60,25))
        elif population[0].quart_tour > 11:
            affiche("Tour : 3/3",(Largeur-60,25))
        else:
            affiche("Tour :"+str(population[0].quart_tour//4+1)+"/3",(Largeur-60,25))
        if population[0].tps_debut == -1 :
            affiche("Chrono : 0s",(Largeur-75,50))
        else :
            affiche("Chrono :"+str(round(population[0].n_frame/100,1))+"s",(Largeur-75,50))
        pygame.display.update()
        clock.tick(SPEED)
    affichage = False
    
# test()
    
while True:
    train()

pygame.quit()
exit()