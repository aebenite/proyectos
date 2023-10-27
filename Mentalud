"""
Created on Wed Oct 25 11:32:03 2023

@author: MENTALUD UES Ixtapaluca, Estado de Mèxico, Mèxico

INTEGRANTES:
    
Miguel Àngel Castillo Martìnez 
Miriam Cortes Niclas 
Litzy Eddaly Grimaldo Pèrez 
Guadalupe Viridiana Velazques Salvador 

ASESORES:

    ING. Armando Escobar Benitez
"""
import cv2
import mediapipe as mp
import math
# =============================================================================
# Realizamos la videocaptura(0)
# =============================================================================
cap=cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,768)

mpDibujo=mp.solutions.drawing_utils
ConfDibu=mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

mpMallaFacial=mp.solutions.face_mesh
MallaFacial=mpMallaFacial.FaceMesh(max_num_faces=1)

while True:
    ret, frame=cap.read()
    
    frameRGB=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    resultados=MallaFacial.process(frameRGB)
    
    px = []
    py = []
    lista = []
    r = 5
    t = 3
    
    if resultados.multi_face_landmarks:
        for rostros in resultados.multi_face_landmarks:
            mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_CONTOURS, ConfDibu, ConfDibu)
            
            for id, puntos in enumerate(rostros.landmark):
                al, an, c=frame.shape
                x, y = int(puntos.x*an), int(puntos.y*al)
                px.append(x)
                py.append(y)
                lista.append([id,x,y])
                if len(lista) == 468:
                    x1, y1 = lista[65][1:]
                    x2, y2 = lista[158][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 
                    cv2.line(frame, (x1,y1),(x2,y2),(0,0,0),t)
                    cv2.circle(frame, (x1,y1), r, (0,0,0), cv2.FILLED)
                    cv2.circle(frame, (x2,y2), r, (0,0,0), cv2.FILLED)
                    cv2.circle(frame, (cx,cy), r, (0,0,0), cv2.FILLED)
                    longitud1 = math.hypot(x2-x1,y2-y1)
                    print(longitud1)
                    
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    cx2, cy2 = (x3+x4) // 2, (y3+y4) // 2
                    cv2.line(frame, (x3,y3),(x4,y4),(0,0,0),t)
                    cv2.circle(frame, (x3,y3), r, (0,0,0), cv2.FILLED)
                    cv2.circle(frame, (x4,y4), r, (0,0,0), cv2.FILLED)
                    cv2.circle(frame, (cx,cy), r, (0,0,0), cv2.FILLED)
                    longitud2 = math.hypot(x4-x3, y4-y3)
                    print(longitud2)
                    
                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    cx3, cy3 = (x5+x6) // 2, (y5+y6) // 2
                    cv2.line(frame, (x5,y5),(x6,y6),(0,0,0),t)
                    cv2.circle(frame, (x5,y5), r, (0,0,0), cv2.FILLED)
                    cv2.circle(frame, (x6,y6), r, (0,0,0), cv2.FILLED)
                    cv2.circle(frame, (cx,cy), r, (0,0,0), cv2.FILLED)
                    longitud3 = math.hypot(x6-x5, y6-y5)
                    print(longitud3)
                    
                    x7, y7 = lista[13][1:]
                    x8, y8 = lista[14][1:]
                    cx4, cy4 = (x7+x8) // 2, (y7+y8) // 2
                    cv2.line(frame, (x7,y7),(x8,y8),(0,0,0),t)
                    cv2.circle(frame, (x7,y7), r, (0,0,0), cv2.FILLED)
                    cv2.circle(frame, (x8,y8), r, (0,0,0), cv2.FILLED)
                    cv2.circle(frame, (cx,cy), r, (0,0,0), cv2.FILLED)
                    longitud4 = math.hypot(x8-x7, y8-y7)
                    print(longitud4)
                    print()
    
                    #Bravo
                    if longitud1 < 27 and longitud2 < 25 and longitud3 > 80 and longitud3 < 160 and longitud4 < 7:
                        cv2.putText(frame, 'Persona enojada', (240,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    
                    #Feliz
                    if longitud1 > 20 and longitud1 < 40 and longitud2 > 10 and longitud2 < 40 and longitud3 > 115 and longitud4 >2 and longitud4 <70:
                        cv2.putText(frame, 'Persona feliz', (240,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
                    
                    #asombrado
                    if longitud1 > 35 and longitud2 > 35 and longitud3 > 90 and longitud3 < 100 and longitud4 > 20:
                        cv2.putText(frame, 'Persona asombrada', (240,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                    
                    #Triste
                    if longitud1 > 20 and longitud1 < 35 and longitud2 > 20 and longitud2 < 35 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame, 'Persona triste', (240,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
    
    cv2.imshow("Reconocimiento de emociones", frame)
    t = cv2.waitKey(1)
    
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
