import pygame
from sys import exit
import math
import time
from info import plot
import random
import json

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

    
def affiche(text, position, color = noir, taille = 20):
    global affichage
    if affichage == True: 
        police=pygame.font.SysFont('arial',taille)
        texte=police.render(text,True,noir)
        zoneTexte = texte.get_rect()
        zoneTexte.center = position
        Zone_jeu.blit(texte,zoneTexte)
        #pygame.display.update() #Mise à jour de la fenêtre

def maximum(liste):
    maxi = liste[0]
    indice = 0
    
    for _ in range(1,len(liste)):
        if maxi < liste[_]:
            maxi = liste[_]
            indice = _
    return (maxi,indice)


class Car :
    def __init__(self,x=837,y=440 ,Q_table1 = {}, Q_table2 = {}, voiture = 'f1.png', voiture_taille = 60):        
        self.voitureLargeur=1920/voiture_taille #450/16
        self.voitureHauteur=1080/voiture_taille #204/16
        self.img=pygame.image.load(voiture) 
        self.img_origine = pygame.transform.scale(self.img, (self.voitureLargeur, self.voitureHauteur))
        
        self.x = x
        self.y = y
        
        self.Q_table1 = Q_table1.copy()
        self.Q_table2 = Q_table2.copy()

        self.acceleration = 0.1
        self.frein = 0.4/3
        self.max_vitesse = 4
        self.frottement = 0.05/3
        self.maniabilité = 2
        self.adherence = 0.05
        
        self.actions = ([1,0],[1,1],[1,2],[0,0],[0,1],[0,2])
        
        self.n_games = 0
        
        self.gamma = 0.99 # long ou court terme
        
        self.dist_max = 0
                                        
    def reset(self):            
        self.reward_tot = 0
        self.img = self.img_origine 
        
        self.angle = 0
        self.position = pygame.Vector2(self.x, self.y)  # Centre logique
        self.tourner(90)

        self.game_over = False
        self.vitesse = 0         # Vitesse actuelle
        self.pos_old = pygame.Vector2(self.position[0],self.position[1])
        self.quart_tour = -1
        self.rotation_mouvement = 0
        self.tps_debut = -1
        self.n_frame = 0
        self.n_games += 1
        self.reward = 0
        self.pause = 0
        self.dist_bord = 2.0
        self.diff_angle = 0
        self.diff_angle_old = 0
        self.distance_parcouru = 0
        self.distance()
        self.reward = 0
        
        self.lr = max(0.0001, 0.001 - self.n_games * 2*10**-7)
        
        self.Jeux()
    
    def distance(self):
        global bord
        
        new_dist = self.distance_parcouru
        dist_min = ((bord[new_dist][0]-self.position[0])**2 + (bord[new_dist][1]-self.position[1])**2)**0.5
        
        i_min = max(0, self.distance_parcouru - 100)
        i_max = min(len(bord), self.distance_parcouru + 200)
        
        for i in range(i_min, i_max):
            if ((bord[i][0]-self.position[0])**2 + (bord[i][1]-self.position[1])**2)**0.5 < dist_min:
                dist_min = ((bord[i][0]-self.position[0])**2 + (bord[i][1]-self.position[1])**2)**0.5
                new_dist = i
            
        # state
        
        self.dist_bord = dist_min
        
        if self.dist_bord > 70:
            self.dist_bord = 7.0
        else :
            self.dist_bord = self.dist_bord//10.0
        
        
        self.diff_angle = (-math.atan2(bord[new_dist+1+int(dist_min)][1]-bord[new_dist+int(dist_min)][1],bord[new_dist+1+int(dist_min)][0]-bord[new_dist+int(dist_min)][0])*180/math.pi+360)%360-self.angle
        self.diff_angle = (self.diff_angle+540)%360-180 # intervalle [-180, 180[
                
        if not map_fini :
            if dist_min < 54 :
                if abs(self.diff_angle) < abs(self.diff_angle_old):
                    self.reward += 1
                else :
                    self.reward -= 1
            else :
                if self.diff_angle < self.diff_angle_old :
                    self.reward += 1
                else :
                    self.reward -= 1
                    
        self.diff_angle_old = self.diff_angle
                
# =============================================================================

        self.diff_angle_devant = (-math.atan2(bord[new_dist+76][1]-bord[new_dist+75][1],bord[new_dist+76][0]-bord[new_dist+75][0])*180/math.pi+360)%360-self.angle
        self.diff_angle_devant = (self.diff_angle_devant+540)%360-180 # intervalle [-180, 180[
        
        
        self.diff_angle_devant = max(-16.0 ,min(15.0, self.diff_angle_devant//6.0)) # -90 à 90; 32 possibilités
        
        # state
        
        self.reward += (new_dist-self.distance_parcouru)
        
        self.distance_parcouru = new_dist

            
    def tourner(self, degre):
        self.angle += degre
        self.angle = self.angle%360
        
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
        
        if (V_new[0]**2 + V_new[1]**2)**0.5 < 0.1:
            self.pause += 1

        else : 
            self.pause = 0
        
        self.position += V_new
        self.rect = self.img.get_rect(center=self.position)
        
    def collision(self, couleur=(12,190,0), x=None, y=None):
        if x == None:
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
        # quart_tour_old = self.quart_tour
        
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

        # if quart_tour_old < self.quart_tour :
        #     self.reward += 50
        # elif quart_tour_old > self.quart_tour :
        #     self.reward -= 50
            
    def Jeux(self):
        global map_fini
        
        if not self.game_over: 
        
            
# =============================== récupération de state ================================================

            vitesse_actuelle = self.position - self.pos_old
                     
            state = ((vitesse_actuelle[0]**2+vitesse_actuelle[1]**2)**0.5//1.01, # 0 à 3 : 4 possibilitées
                     self.dist_bord,  # 0 à 7 : 8 possibilitées
                     self.diff_angle_devant)  # -16 à 15 : 32 possibilitées
            # total : 4*8*32 = 1024 possibilitées
            

            if state not in self.Q_table1 :
                self.Q_table1[state] = [0 for i in range(6)]
            if state not in self.Q_table2 :
                self.Q_table2[state] = [0 for i in range(6)]
                
# =============================== récupération de l'action =============================================
                    

           # if abs(self.dist_max - self.distance_parcouru) < 30 and not map_fini :
            #    epsilon = 50
            #else : 
             #   epsilon = 2
            
            epsilon = max(max(3,30-self.n_games//100),100-self.n_games//10) #TODO
            
            if random.randint(0,100) > epsilon :
                action = maximum([self.Q_table1[state][a]+self.Q_table2[state][a] for a in range(6)])[1]
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
            self.reward = 0
            
            if not map_fini and (vitesse_actuelle[0]**2+vitesse_actuelle[1]**2)**0.5 > 2.5 and avance == True:
                self.reward -= 2
                
                
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
            
            if self.collision() or self.pause > 100:
                self.game_over = True
                
                self.reward -= 200
                
                if self.distance_parcouru > self.dist_max :
                    self.dist_max += (self.distance_parcouru-self.dist_max)*0.01
                    #print(self.dist_max)


            self.tour()
            
            if self.quart_tour>11:
                self.tps = str(round(time.time()-self.tps_debut,1))
                self.game_over = True
                self.dist_max = (self.distance_parcouru-self.dist_max)*0.01 # comme il a fini il a distance_parcouru >= dist_max
            #print(state,self.actions[action],self.reward)
            
# =============================== mise à jour de la Q_table ============================================

            vitesse_actuelle = self.position - self.pos_old
                            
            state_new = ((vitesse_actuelle[0]**2+vitesse_actuelle[1]**2)**0.5//1.01,
                     self.dist_bord,
                     self.diff_angle_devant)

            if state_new not in self.Q_table1 :
                self.Q_table1[state_new] = [0 for i in range(6)]
            if state_new not in self.Q_table2 :
                self.Q_table2[state_new] = [0 for i in range(6)]
                
            if random.randint(0,1)%2 == 0:
                self.Q_table1[state][action] += self.lr * (self.reward + self.gamma * self.Q_table2[state_new][maximum(self.Q_table1[state_new])[1]] - self.Q_table1[state][action])
            else :
                self.Q_table2[state][action] += self.lr * (self.reward + self.gamma * self.Q_table1[state_new][maximum(self.Q_table2[state_new])[1]] - self.Q_table2[state][action])

            self.reward_tot += self.reward
            self.n_frame += 1

def isbord(x, y, couleur=(95,207,43)):   
    if not (0 <= x < Largeur and 0 <= y < Hauteur): # hors de l'écran
        return True

    pixel = terrain.get_at((x, y)) # donne la couleur du pixel
    
    count = 0
    for c in range(3):
        if couleur[c]-30<pixel[c]<couleur[c]+30:
            count+=1
    if count == 3:
        return True
    return False

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
        
    return tuple(bord[-100:]+bord+bord+bord+bord[:200]) # il faut une marge | bord[-100:] la voiture commence avant la ligne d'arrivée

bord = centre_piste()

def save(Q_table, filename="save.json"):
    try:
        with open(filename, "w") as f:
            json.dump({"Q_table": list(Q_table.items())}, f)
    except Exception as e:
        print("❌ Erreur lors de la sauvegarde :", e)
        
def load(filename="save.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        Q_table = {tuple(k): v for k, v in data["Q_table"]}
        print(f"✅ Q-table chargée depuis {filename}")
        return Q_table
    except Exception as e:
        print("❌ Erreur lors du chargement de la Q-table :", e)
        return {}
    
    
affichage = False #TODO
map_fini = False

car = Car()

sauvegarde1 = load(filename="save1.json")
sauvegarde2 = load(filename="save2.json")

if sauvegarde1 != None and sauvegarde2 != None:
    car=Car(Q_table1=sauvegarde1, Q_table2=sauvegarde2)
    


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
    if car.n_games%20==0:
        affichage = True
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
            else:
                affiche("Tour :"+str(car.quart_tour//4)+"/3",(Largeur-60,25))
            if car.tps_debut == -1 :
                affiche("Chrono : 0s",(Largeur-75,50))
            else :
                affiche("Chrono :"+str(round(car.n_frame/100,1))+"s",(Largeur-75,50))
            
            pygame.display.update() #on rafraichit l'écran.
            clock.tick(0)
        
    if car.quart_tour>11 or map_fini == True:
        map_fini = True
        # if car.n_games < 502 :
        #     map_fini = False
        if car.quart_tour > 11:
            scores_plot.append(round(car.n_frame/100,2))
            print("temps : "+str(round(car.n_frame/100,2)))
            if car.n_frame/100 < record:
                record = car.n_frame/100
            print("record : "+str(round(record,2))+"s")
        else :
            print("fail, distance : "+str(round(car.distance_parcouru,2)))
            print("record : "+str(round(record,2))+"s")
    else :
        scores_plot.append(round(car.distance_parcouru,2))
        print("distance : "+str(round(car.distance_parcouru,2)))
    print("taille Q table : "+str(len(car.Q_table1)))
    print()
    reward_moy.append(car.reward_tot)
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
        save(car.Q_table1, filename="save1.json")
        save(car.Q_table2, filename="save2.json")
        
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