######### codigo  Paralelo ########################

from FastMinDegree_grafoOpt import grafo, segundos_a_segundos_minutos_y_horas
from AGEC_CreacionPoblacionOp import Individuo, Poblacion
from glib import gphs
import numpy as np
import time
import pickle
import os
from os import mkdir
from copy import deepcopy
from Controlmutacion import mutacionF

def Al_evol_N(name: str, mut_n, P0, G, G_fitmin, mutprob, epoc, 
              rounds, parejas, mutacion, ep_ext: int=30, caja: bool=False, 
              seleccion: str='Ruleta', M_max: float=15, MFija=False, ver: bool=True):
    t_AG = time.time()
    P = deepcopy(P0)
    fitMin = np.array([], dtype=int)
    fitMax = np.array([], dtype=int)
    t = time.time()
    R = '\n'
    if caja:
        dcaja = np.zeros((6, P0.tamaño))
    n = 0
    k1 = 0
    r = 0
    # print(P.mut[0])
    
    llego = False
    l = 0
    while(P.fit[0] >= G_fitmin and n < epoc):
        if P.fit[0] == G_fitmin: 
            if l == 0:
                llego = True
                P_min = deepcopy(P)
                n_min = deepcopy(n)
                R_min = deepcopy(R)
            l += 1
        
        if l > ep_ext:
            P = deepcopy(P_min)
            n = n_min
            R = R_min
            break
        
        mut_p = P.mut[0]
        P.MutList = np.append(P.MutList, mut_p)
        P.NewP(q=rounds, k=parejas, mutacion=mutacion, g=G, seleccion=seleccion)
        P.fit_MinMax(todos=True)
        P.fit_std = np.std(P.fitList)
        
        if P.fit_std == 0: 
            k2 = k1
            k1 = n
            if k2 == (n-1):
                r += 1
                if r == 100:
                    R += 'Se obtuvo desviación estándar cero por 100 épocas seguidas'
                    break
            else:
                r = 0
                
        fitMin = np.append(fitMin, [P.fit[0]])
        fitMax = np.append(fitMax, [P.fit[1]])
        # print(fitMin)
        if n==0:
            if caja:
                dcaja[0] = P.fitList
            T = time.time()- t
            R += F'Epoc: {n+1}; Mutación: {mut_p*100}%\nMáximo fitness obtenido: {P.fit[1]}\nMínimo fitness obtenido: {P.fit[0]}\n'
            R += F'Tiempo: {segundos_a_segundos_minutos_y_horas(T)}\n\n'
            if ver:
                print(F'Epoc: {n+1}; Mutación: {mut_p*100}%\nMáximo fitness obtenido: {P.fit[1]}\nMínimo fitness obtenido: {P.fit[0]}')
                print(F'Tiempo: {segundos_a_segundos_minutos_y_horas(T)}\n')
        
        if caja and n<50:
            if (n+1)%10 == 0:
                dcaja[int((n+1)/10)] = P.fitList
        
        if (n+1)%100 == 0:
            
            T = time.time()- t
            R += F'Epoc: {n+1}; Mutación: {mut_p*100}%\nMáximo fitness obtenido: {P.fit[1]}\nMínimo fitness obtenido: {P.fit[0]}\n'
            R += F'Tiempo: {segundos_a_segundos_minutos_y_horas(T)}\n\n'
            if ver:
                print(F'Epoc: {n+1}; Mutación: {mut_p*100}%\nMáximo fitness obtenido: {P.fit[1]}\nMínimo fitness obtenido: {P.fit[0]}')
                print(F'Tiempo: {segundos_a_segundos_minutos_y_horas(T)}\n')
            t = time.time()
        n += 1
        if not MFija:
            mut_p = mutacionF(P.fit_std, mut_n, n+1, M_max=M_max)/100
            P.mut = [mut_p, 1- mut_p]
    # print(fitMin, fitMax)
    
        
    P.indices = np.where(P.fitList==P.fit[0])[0][0]
    P.tiempo[0] = time.time() - t_AG
    P.fitMinMax = [fitMin, fitMax]
    
    if caja:
        return P, R, n, llego, dcaja
    return P, R, n, llego


def Resultados_par(j, P, epoc, rounds, parejas, mutacion, seleccion, Mfija = True):
    
    Resultado = F'\nResultado de Algoritmo genético para Exp_{j}\n'
    
    Resultado += F'Tamaño de la poblacion {P.tamaño}, Parejas {parejas}, rounds {rounds}, epocas {epoc}\n\n'
    
    if Mfija:
        Resultado += F'mutación fija: {mutacion} {P.mut[0]*100}%\n'
    else:
        Resultado += F'mutación variable: {mutacion}\n'
    Resultado += F'Método de selección: {seleccion}\n'
    Resultado += F'Porcentaje de indiviuos aleatorios en la población inicial: {P.porcenP[0]*100}%\n'
    Resultado += F'tiempo de creación de población: {segundos_a_segundos_minutos_y_horas(P.tiempo[1])}\n'
    Resultado += F'tiempo de ejecución: {segundos_a_segundos_minutos_y_horas(P.tiempo[0])}\n'
    # print(P.fit)
    Resultado += F'Máximo fitness obtenido: {P.fit[1]}\n'
    Resultado += F'Mínimo fitness obtenido: {P.fit[0]}\n'
    
    # Resultado += 'Orden de menor fitness: \n ['
    # for i in range(len(P.indiv[P.indices].orden.orden)):
    #     Resultado += F'{P.indiv[P.indices].orden.orden[i]}, '
    # Resultado += ']\n'
    P.fit[0], P.fit[1] = P.fitMinMax[1][-1], P.fitMinMax[0][-1]
    return Resultado

def Archivo_minfit(G_fitmin, ep_fitmin, name, FMD = False, FMD_R=''):
    if not os.path.isdir(F'Case{G_fitmin.ext.tamaño}/'):
        mkdir(F'Case{G_fitmin.ext.tamaño}/')
    
    if not os.path.isdir(F'Case{G_fitmin.ext.tamaño}/{name}'):
        mkdir(F'Case{G_fitmin.ext.tamaño}/{name}')
        
    with open(F"Case{G_fitmin.ext.tamaño}/{name}/G_fitmin.txt", "wb") as file:
        pickle.dump([G_fitmin, ep_fitmin], file)
    
    if FMD:
        with open(F"Case{G_fitmin.ext.tamaño}/G_fmd.txt", "wb") as file:
            pickle.dump([G_fitmin, FMD_R], file)
    

def Creararchivo_par(P, epocas, j, name):

    if not os.path.isdir(F'Case{P.N_nodos}/'):
        mkdir(F'Case{P.indiv[0].ext.tamaño}/')
    
    if not os.path.isdir(F'Case{P.N_nodos}/{name}'):
        mkdir(F'Case{P.N_nodos}/{name}')
    
    Ext_c = [P.fit[0], P.fit[1], epocas, np.std(P.fitList), 
             np.mean(P.fitList), P.fitList, P.indiv[P.indices]]
    
    i = True
    while(i):
        if os.path.isfile(F'Case{P.N_nodos}/{name}/Ext_{j}.txt'):
            # print(j)
            j += 1
        if not os.path.isfile(F'Case{P.N_nodos}/{name}/Ext_{j}.txt'):
            with open(F"Case{P.N_nodos}/{name}/Ext_{j}.txt", "wb") as file:
                pickle.dump(Ext_c, file)
            i = False
    
    return j

def Poblacion_AEPar(mut_n: int, m: int, name: str, N: int, M_max, MFija, Tamaño, 
                 G_fit, epoc, ep_ext: int=30, parejas: int=50, rounds: int=5) -> None:
    Tiempo = time.time()
    try:
        print(True)
        with open(F'Case{m}/Grafo.txt', "rb") as f:
            G = pickle.load(f)
    except FileNotFoundError:
        t = time.time()
        G = grafo(name, tamaño = m, edges = gphs[F'{name}.m'], ML='listaEdges')
        if not os.path.isdir(F'Case{m}/'):
            mkdir(F'Case{m}/')
        with open(F'Case{m}/Grafo.txt', "wb") as file:
            pickle.dump(G, file)
        print(F'Creación de grafo: {segundos_a_segundos_minutos_y_horas(time.time() - t)}')
            
    # inicial = np.array([i for i in range(G.tamaño)], dtype=int)
    inicial = np.arange(G.tamaño, dtype=int)
    
    mutprob=(mut_n, 1- mut_n)
    porcenP= (1, False)
    
    #### Nombre del archivo: 'tamaño_parejas_rounds_epoc_MinimoObtenido_%mut'
    if MFija:
        nombre =F'{G.tamaño}_{Tamaño}_{parejas}_{rounds}_{epoc}_MutacionFija{int(mut_n*100)}'
    else:
        nombre =F'{G.tamaño}_{Tamaño}_{parejas}_{rounds}_{epoc}_MutacionVariable{int(M_max)}'
        
        
    # if os.path.isfile(F'Case{m}/Grafo_fmd.txt'):
    try:
        with open(F'Case{m}/Grafo_fmd.txt', "rb") as f:
            fmd = pickle.load(f)
            G_fmd, FMD_R = fmd[0], fmd[1]
        print(True)
    # else:
    except FileNotFoundError:
        t_FMD = time.time()
        FMD_R = 'Fast Min Degree\n'
        G_fmd = Individuo(grafo=G, Metodo='FMD', inicial=inicial)
        FMD_R += F'Tiempo de ejecución: {segundos_a_segundos_minutos_y_horas(time.time() - t_FMD)} \n'
        FMD_R += F'Cantidad de aristas con algoritmo de grado mínimo: {G_fmd.fit} \n\n'
    
        if not os.path.isdir(F'Case{m}/'):
            mkdir(F'Case{m}/')
        Archivo_minfit(G_fmd, 0, nombre, FMD = True, FMD_R=FMD_R)
    
    
    try:
        with open(F'Case{m}/PoblacionInicial{Tamaño}.txt', "rb") as f:
            P_inicial = pickle.load(f)
    except FileNotFoundError:
        t_AG = time.time()
        P_inicial = Poblacion(inicial=inicial, N=Tamaño, g = G, mutprob=mutprob, porcenP=porcenP, G_fmd= G_fmd)
        P_inicial.tiempo[1] = time.time() - t_AG
        # mkdir(F'Case{m}/')
        with open(F'Case{m}/PoblacionInicial{Tamaño}.txt', "wb") as file:
            pickle.dump(P_inicial, file)    
    
    P_inicial.mut = mutprob
    
    FMD_R += F'tiempo de creación de población: {segundos_a_segundos_minutos_y_horas(P_inicial.tiempo[1])}\n'  #11.661226097742716 minutos
    # print(FMD_R)
    seleccion = 'Ruleta'
    
    mutacion = 'SwapMutation'  #'ScrambleMutation'  #
    # epoc = 400
    j = 0
    
    if os.path.isfile(F'Case{m}/{name}/G_fitmin.txt'):
        with open(F'Case{m}/{name}/G_fitmin.txt', "rb") as f:
            mfit = pickle.load(f)
            G_fitmin = mfit[0]
            
        
    else:
        G_fitmin = G_fmd
    
    if G_fmd.fit < G_fitmin.fit:
            G_fitmin = G_fmd     
        
    R = ' '
    for _ in range(N):
        #Al_evol_N(name, mut_n, P0, G, G_fitmin, mutprob, epoc, 
                      # rounds, parejas, mutacion, caja=False, 
                      # seleccion='Ruleta', M_max=15, MFija=False, ver=True):
        Pob, _, epocas, llego = Al_evol_N(name, mut_n, P_inicial, G, 
                                      G_fitmin.fit, mutprob, epoc,  rounds, 
                                      parejas, mutacion, caja=False, 
                                      seleccion=seleccion, ep_ext=ep_ext, M_max=M_max, 
                                      MFija=MFija, ver=False)
        
        if Pob.indiv[Pob.indices].fit < G_fitmin.fit:
            G_fitmin = Pob.indiv[Pob.indices]
            Archivo_minfit(G_fitmin, epocas, nombre)
            
        # print(epocas)
        # Creararchivo_par(P, epocas, j, name)
        j = Creararchivo_par(Pob, epocas, j, nombre)
        
        Resultado = Resultados_par(j, Pob, epocas, rounds, 
                              parejas,  mutacion,
                              seleccion=seleccion, Mfija=MFija)
        R += Resultado
        # print(Resultado)
        j += 1 
        # print(F'Menor fitness encontrado: {G_fitmin.fit}')
    
    with open(F"Case{m}/{nombre}/Resultados{j}.txt","w") as file: 
            file.write(FMD_R + R)
            
    print(F'Tiempo de ejecución de {name} en Resultados{j}: {segundos_a_segundos_minutos_y_horas(time.time() - Tiempo)}')
             

                
if __name__ == "__main__":
    
    m = [39, 14, 300, 500, 300, 118, 500, 300 ]
    name = ['case39', 'case14', 'case300', 'pglib_opf_case500_goc', 'case300', 'case118', 'pglib_opf_case500_goc', 'case300']
    G_fit = [False, 661, 1220, 661, 265, 1220, 661]
    
    #Poblacion_AGPar(mut_n, m, name, N, M_max, MFija, Tamaño, 
                     # G_fit, epoc, parejas=50, rounds=5)
            
            # MFija = True - Mutacion fija
            # MFija = False - Mutacion variable con maximo M_max
            # N = camtidad de veces que se ejecuta el algoritmo
            # Tamaño = tamaño de poblacion
            # epoc = maximo de epocas en el algoritmo
    Poblacion_AEPar(0.15, m[0], name[0], 10, 20, MFija=True, 
                    Tamaño=20, G_fit=G_fit[0], epoc = 20, parejas=10, rounds=5)
    # Valores originales: (0.15, m[0], name[0], 10, 20, MFija=True, 
    #                       Tamaño=200, G_fit=G_fit[0], epoc = 200, parejas=100, rounds=5)

    