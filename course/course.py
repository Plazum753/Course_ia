import pygame
from sys import exit
import math
import time
import json

game_play = [] #TODO


def save(individu, filename="save.json"):
    try:
        with open(filename, "w") as f:
            json.dump(individu, f)
    except Exception as e:
        print("❌ Erreur de sauvegarde :", e)

pygame.init()

vert=(0,180,0)
noir=(0,0,0)
blanc = (255,255,255)

Largeur = 976
Hauteur = 912

terrain = pygame.image.load('Map.png') 
terrain = pygame.transform.scale(terrain, (Largeur, Hauteur))

SPEED = 100

Zone_jeu = pygame.display.set_mode((Largeur,Hauteur))
pygame.display.set_caption("Course") #titre de la fenêtre


def rejouerOUquitter():
     for event in pygame.event.get([pygame.KEYDOWN,pygame.KEYUP,pygame.QUIT]): #si l'évènement est touche appuyé, relaché ou quit
         if event.type == pygame.QUIT :
             pygame.quit()
             exit()
         elif event.type == pygame.KEYUP : #touche relachée
             continue
         return 1 #event.key #on renvoie quelque chose différent de none
     return None


def gameOver():    
    pygame.display.update() #Mise à jour de la fenêtre
    
    while rejouerOUquitter()==None:
        pass
    car.reset() #Dès qu'on sort de la while, on relance le jeu
    
def affiche(text, position, color = noir, taille = 20):
    police=pygame.font.SysFont('arial',taille)
    texte=police.render(text,True,noir)
    zoneTexte = texte.get_rect()
    zoneTexte.center = position
    Zone_jeu.blit(texte,zoneTexte)
    pygame.display.update() #Mise à jour de la fenêtre

    
class Car :
    def __init__(self):
        self.voitureLargeur=450/16
        self.voitureHauteur=204/16
        self.clock = pygame.time.Clock()
        self.img=pygame.image.load('voiture.png') 
        self.img_origine = pygame.transform.scale(self.img, (self.voitureLargeur, self.voitureHauteur))
        
        self.acceleration = 0.1
        self.frein = 0.4/3
        self.max_vitesse = 4
        self.frottement = 0.05/3
        self.maniabilité = 2
        self.adherence = 0.05
        
    def reset(self):
        self.img = self.img_origine 
        self.position = pygame.Vector2(837, 440)  # Centre logique
        self.angle = 0
        self.tourner(90)
        self.game_over = False
        self.vitesse = 0         # Vitesse actuelle
        self.pos_old = pygame.Vector2(self.position[0],self.position[1])
        self.quart_tour = -1
        self.rotation_mouvement = 0
        self.tps_debut = -1
        
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
        if vitesse_actuelle.length() > 4:
            while True :
                pass
        
        self.pos_old = pygame.Vector2(self.position[0],self.position[1])
        
        V_new = vitesse_actuelle + self.adherence*(direction * self.vitesse - vitesse_actuelle)
        
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
    
    def Jeux(self):
        
        global game_play #TODO
        
        while not self.game_over: #on demarre le jeu
            for event in pygame.event.get():
                if event.type == pygame.QUIT : # lorsqu'on clique sur la croix rouge
                    self.game_over=True
                    pygame.quit()
                    exit()
            bouton = [0,0,0]
            keys = pygame.key.get_pressed()
            avance = keys[pygame.K_UP]
            recule = keys[pygame.K_DOWN]
            droite = keys[pygame.K_RIGHT]
            gauche = keys[pygame.K_LEFT]
                
            if droite and not gauche:
                self.rotation_mouvement = -self.maniabilité
                bouton[2] = 1
            elif gauche and not droite:
                self.rotation_mouvement = +self.maniabilité
                bouton[1] = 1
            else:
                self.rotation_mouvement = 0

            # Accélération et freinage
            if avance:
                self.vitesse += self.acceleration
                bouton[0] = 1
                
            # else : # TODO à enlever plus tard
            #     self.vitesse -= self.frein
            # game_play.append(bouton.copy())

            if recule:
                self.vitesse -= self.frein #TODO à remettre plus tard
            if not recule and not avance:
                # Inertie : ralentit progressivement
                if self.vitesse > 0:
                    self.vitesse -= self.frottement
                    if self.vitesse < 0:
                        self.vitesse = 0
                elif self.vitesse < 0:
                    self.vitesse += self.frottement
                    if self.vitesse > 0:
                        self.vitesse = 0
            
            # Limitation de la vitesse maximale
            if self.vitesse > self.max_vitesse:
                self.vitesse = self.max_vitesse
            if self.vitesse < 0 :#-self.max_vitesse / 2:
                self.vitesse = 0 #-self.max_vitesse / 2  # Recul plus lent
                
            if abs(self.vitesse) > 0.2:
                #rotation_mouvement *= (self.vitesse / self.max_vitesse)
                self.tourner(self.rotation_mouvement)
                if self.tps_debut == -1 :
                    game_play = [[1,0,0],[1,0,0],[1,0,0]]
                    self.tps_debut = time.time()
                
            self.avancer()
            
            Zone_jeu.blit(terrain,(0, 0)) 
            Zone_jeu.blit(self.img,self.rect.topleft) #on superpose la voiture dans la zone de jeu
            
            if self.collision((12,190,0)):
                affiche("Game Over",(Largeur/2,((Hauteur)/2-50)),taille = 100)
                return gameOver()
            self.tour()
            # if self.quart_tour<0:
            #     affiche("Tour : 0/3",(Largeur-60,25))
            # else:
            #     affiche("Tour :"+str(self.quart_tour//4)+"/3",(Largeur-60,25))
            # if self.tps_debut == -1 :
            #     affiche("Chrono : 0s",(Largeur-75,50))
            # else :
            #     affiche("Chrono :"+str(round(time.time()-self.tps_debut,1))+"s",(Largeur-75,50))
            
            if self.quart_tour>11:
                tps = str(round(time.time()-self.tps_debut,1))
                affiche("Fini en"+tps+"s",(Largeur/2,((Hauteur)/2-50)),taille = 100)
                save(game_play)
                return gameOver()
    
            pygame.display.update() #on rafraichit l'écran.
            self.clock.tick(SPEED)

car = Car()
car.reset() #on lance le jeu
pygame.quit() #on sort de la boucle donc on quitte
exit()