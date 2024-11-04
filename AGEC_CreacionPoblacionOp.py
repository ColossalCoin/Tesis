######### codigo 3 ########################

import numpy as np
from FastMinDegree_grafoOpt import grafo, segundos_a_segundos_minutos_y_horas
from copy import deepcopy
from AGEC_TiposMutacionOp import Orden
import time
from glib import gphs
from os import mkdir
import os
import pickle

class Individuo:
    def __init__(self, orden: list[int]=[], grafo: list=[], inicial: list=[], ver: bool=False, Metodo: str='Orden'):
        """
        

        Parameters
        ----------
        orden : list[int], optional
            Una lista que representa el orden en el que se procesan o visitan los nodos del grafo. 
            Este orden define la solución propuesta por el individuo. The default is [].
        grafo : list, optional
            DESCRIPTION. The default is [].
        inicial : list, optional
            DESCRIPTION. The default is [].
        ver : bool, optional
            DESCRIPTION. The default is False.
        Metodo : str, optional
            Una cadena que especifica el método de eliminación utilizado para construir el individuo. 
            Puede ser 'FMD' u 'Orden'. The default is 'Orden'.

        Returns
        -------
        None.

        """
        if Metodo == 'FMD':
            self.ext, self.g_aux, orden = grafo.Extension(Eliminacion=Metodo)
        else:
            self.ext, self.g_aux, orden = grafo.Extension(Eliminacion=Metodo, orden=orden)
        self.orden = Orden(inicial=inicial, orden=orden)
        self.fit = self.ext.len_edges
        

class Poblacion:
    def __init__(self, g: grafo, inicial: list[int]=[], N: int=100, mutprob: tuple[float]=(0.05, 0.95), 
                 G_fmd: list=[], porcenP: tuple=(1, False), ver: bool=False):
        """
        Esta clase representa una población de individuos en un algoritmo evolutivo.

        Parameters
        ----------
        g : grafo
            Grafo utilizado para evaluar a los individuos.
        inicial : list[int], optional
            Una lista que representa el orden inicial para los individuos. The default is [].
        N : int, optional
            El tamaño de la población (número de individuos). The default is 100.
        mutprob : tuple[float], optional
            Una lista con dos valores que representan las probabilidades de mutación mínima y máxima.
            The default is [0.05, 0.95].
        G_fmd : list, optional
            Una lista que presumiblemente contiene el mejor individuo conocido (FMD). The default is [].
        porcenP : tuple, optional
            Una tupla que define el porcentaje de la población inicial a generar aleatoriamente y un booleano 
            que indica si se debe incluir el FMD en la población inicial. The default is (1, False).
        ver : bool, optional
            Un indicador para habilitar la impresión de información durante la inicialización 
            (True para imprimir, False para omitir). The default is False.

        Returns
        -------
        None.

        """
        self.N_nodos = g.tamaño
        self.indiv = [[] for _ in range(N)]
        self.tamaño = N
        self.mut = mutprob
        self.MutList = np.array([])
        self.gphdif = []
        self.porcenP = porcenP
        self.FMD = G_fmd
        
        N_p = int(N*self.porcenP[0])
        if ver:
            t = time.time()
            print('inicio')
            print(F'Cantidad de población aleatoria: {N_p}')
        
        self.fitList = np.zeros(self.tamaño, dtype=int)
        for i in range(N_p):  
            if ver:
                t_ = time.time() - t
                print(F'individuos creados: {i}, tiempo: {segundos_a_segundos_minutos_y_horas(t_)}')
            o = deepcopy(inicial)
            np.random.shuffle(o)
            self.indiv[i] = Individuo(orden=o, grafo=g, inicial=inicial)
            self.fitList[i] = self.indiv[i].fit
        if N_p < N:
            self.indiv[N_p] = deepcopy(self.FMD)
            self.fitList[N_p] = self.indiv[N_p].fit
            for i in range(N_p+1, N):
                if ver:
                    t_ = time.time() - t
                    print(F'individuos creados: {i}, tiempo: {segundos_a_segundos_minutos_y_horas(t_)}')
                if not porcenP[1]:
                    self.indiv[i] = deepcopy(self.FMD)
                    self.fitList[i] = self.indiv[i].fit
        self.fit = np.zeros(2, dtype=int)
        self.fit_MinMax(todos=True) 
        self.prob = self.Select_prob(todos=True)
        # F = self.fitCalcula()
        self.fit_std = np.std(self.fitList)
        self.indices = []
        self.tiempo = [0,0]
        self.fitMinMax = []
        
    
    def fit_MinMax(self, indices: list=[], todos: bool=False) -> float: 
        if todos:
            # for i in range(self.tamaño):
            #     self.fitList[i] = self.indiv[i].fit
            self.fit[0], self.fit[1] = self.fitList.min(), self.fitList.max()
        else:
            f = []
            for i in indices:
                f.append(self.indiv[i].fit)
            return min(f), max(f)

    def Select_prob(self, indices=[], todos=False):
        if todos:
            f = (self.fitList.max() - self.fitList +1)
            f = f/f.sum()
            return f
        else:
            f = self.fitList[indices]
            f = (max(f) - np.array(f) +1).tolist()
            f = (np.array(f)/sum(f)).tolist()
            return f  
    
    def Select_probRanking(self):
        f = self.fitList
        orden = f.argsort()
        orden = orden.argsort()
        prob = orden.max() - orden +1
        return prob/prob.sum()
        
    
    def Ranking(self, k, mutacion, g, ver=False):
        offspring = []
        if 2*k == self.tamaño:
            par = np.random.choice(self.tamaño, size=(k, 2), replace=False)
        else:
            self.prob = self.Select_probRanking()
            par = np.random.choice(self.tamaño, size=(k, 2), replace=False, p=self.prob)
        for i in range(k):
            p1 = self.indiv[par[i][0]].orden
            p2 = self.indiv[par[i][1]].orden
            if ver:
                print('Pareja a cruzar')
                print('p1 =', par[i][0], p1.orden, self.indiv[par[i][0]].fit)
                print('p2 =', par[i][1], p2.orden, self.indiv[par[i][1]].fit)
            hijo_or = p1.PMX(p2)
            q = np.random.choice(2, p=self.mut)
            if ver:
                print('Mutacion=', mutacion, q)
            if q:
                eval("hijo_or." + mutacion + "()")
            offspring.append(Individuo(hijo_or.orden, g, self.indiv[0].orden.inicial))
        return offspring
    
    def Ruleta(self, k, mutacion, g, ver=False):
        offspring = []
        if 2*k == self.tamaño:
            par = np.random.choice(self.tamaño, size=(k, 2), replace=False)
        else:
            self.prob = self.Select_prob(todos = True)
            par = np.random.choice(self.tamaño, size=(k, 2), replace=False, p=self.prob)
        for i in range(k):
            p1 = self.indiv[par[i][0]].orden
            p2 = self.indiv[par[i][1]].orden
            if ver:
                print('Pareja a cruzar')
                print('p1 =', par[i][0], p1.orden, self.indiv[par[i][0]].fit)
                print('p2 =', par[i][1], p2.orden, self.indiv[par[i][1]].fit)
            hijo_or = p1.PMX(p2)
            q = np.random.choice(2, p=self.mut)
            if ver:
                print('Mutacion=', mutacion, q)
            if q:
                eval("hijo_or." + mutacion + "()")
            offspring.append(Individuo(hijo_or.orden, g, self.indiv[0].orden.inicial))
        return offspring
    
    def NewP(self, q, k, mutacion, g, ver=False, seleccion='Ruleta'):
        offspring = eval("self." + seleccion + "(k, mutacion=mutacion, g=g)")
        surv, surv_off = self.R_r_Tournament(q, k, offspring, ver2=ver)
        if ver:
            print(surv_off)
        for i in range(self.tamaño):
            if i not in surv:
                if ver:
                    print(i)
                self.indiv[i] = offspring[surv_off[0]-self.tamaño]
                self.fitList[i] = self.indiv[i].fit
                surv_off.pop(0)
    
    def R_r_fit(self, offspring, s):
        if s < self.tamaño:
            return self.indiv[s].fit
        else:
            return offspring[s-self.tamaño].fit
            
    def R_r_Tournament(self, q: int, l: int, offspring, ver1: bool=False, ver2: bool=False) -> tuple(list, list):
        """
        Implementa el algoritmo de torneo R_r para selección.

        Parameters
        ----------
        q : int
            El número de rondas en el torneo.
        l : int
            El número de descendientes a seleccionar.
        offspring : TYPE
            La lista de descendientes.
        ver1 : bool, optional
            DESCRIPTION. The default is False.
        ver2 : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        tuple(list, list)
            Una tupla con dos listas. La lista de individuos sobrevivientes y 
            la lista de descendientes sobrevivientes.

        """
        t: list = [i for i in range(self.tamaño+l)]
        wins: np.array = np.zeros(self.tamaño+l)
        surv: list = []
        surv_off: list= []
        
        for s1 in range(self.tamaño+l):
            t_s = t.copy()
            t_s.pop(s1)
            for _ in range(q):
                s2 = np.random.choice(t_s)
                t_s.pop(t_s.index(s2))
                s1_ = self.R_r_fit(offspring, s1)
                s2_ = self.R_r_fit(offspring, s2)
                if ver1:
                    print('s1', s1, s1_)
                    print('s2', s2, s2_)
                if s1_ > s2_:
                    wins[s2] += 1
                else:
                    wins[s1] += 1
                if ver1:
                    print('wins', wins)
        wins = wins.tolist()
        for _ in range(self.tamaño):
            if ver2:
                print('t', t)
                print('wins', wins)
                print('surv', surv)
            s = wins.index(max(wins))
            surv.append(t[s])
            if t[s] > self.tamaño-1:
                surv_off.append(t[s])
            t.pop(s)
            wins.pop(s)
        # surv.sort()
        if ver2:
            print('surv', surv)
            print('surv_off', surv_off)
        return surv, surv_off
    
    def fitCalcula(self):
        f = np.zeros(self.tamaño)
        for i in range(self.tamaño):
            f[i] = self.indiv[i].fit
        return f


def Function() -> int:
    G_fmd_118 = grafo(edges=gphs['case118.m'], ML='listaEdges')
    # inicial = np.array([i for i in range(G_fmd_118.tamaño)])
    inicial = np.arange(G_fmd_118.tamaño, dtype=int)
    T = np.zeros(20)
    for i in range(20):
        t = time.time()
        G_Ext = G_fmd_118.Extension(orden = inicial)
        T[i] = time.time() - t
    print(T.sum())
    return T.sum()


if __name__=="__main__":
    m_0 = [118,500, 300 ]
    name_0 = [ 'case118', 'pglib_opf_case500_goc', 'case300']
    G_fit_0 = [ 265, 1220, 661]
    
    mut_n = 10*m_0[0] 
    m = m_0[0] 
    name = name_0[0] 
    N = 1
    M_max = [20] 
    MFija=False
    Tamaño=200 
    G_fit=G_fit_0[0]
    parejas=100 
    rounds=5
    
    if os.path.isfile(F'Case{m}/Grafo.txt'):
        print(True)
        with open(F'Case{m}/Grafo.txt', "rb") as f:
            G = pickle.load(f)
    else:
        t = time.time()
        G = grafo(name, tamaño = m, edges = gphs[F'{name}.m'], ML='listaEdges')
        if not os.path.isdir(F'Case{m}/'):
            mkdir(F'Case{m}/')
        with open(F'Case{m}/Grafo.txt', "wb") as file:
            pickle.dump(G, file)
        print(F'Creación de grafo: {segundos_a_segundos_minutos_y_horas(time.time() - t)}')
    
    inicial = np.array([i for i in range(G.tamaño)], dtype=int)
    
    if MFija:
        mut_p = M_max[0]/100
    else:
        mut_p = 0.1
    mutprob=[mut_p, 1- mut_p]
    porcenP= [1, False]
    
    t_FMD = time.time()
    FMD_R = 'Fast Min Degree\n'
    G_fmd = Individuo(grafo=G, Metodo='FMD', inicial=inicial)
    FMD_R += F'Tiempo de ejecución: {segundos_a_segundos_minutos_y_horas(time.time() - t_FMD)} \n'
    FMD_R += F'Cantidad de aristas con algoritmo de grado mínimo: {G_fmd.fit} \n\n'
    
    if os.path.isfile(F'Case{m}/PoblacionInicial{Tamaño}_{M_max[0]}.txt'):
        with open(F'Case{m}/PoblacionInicial{Tamaño}_{M_max[0]}.txt', "rb") as f:
            P_inicial = pickle.load(f)
    else:
        t_AG = time.time()
        P_inicial = Poblacion(inicial=inicial, N=Tamaño, g = G, mutprob=mutprob, porcenP=porcenP, G_fmd= G_fmd)
        P_inicial.tiempo[1] = time.time()-t_AG
        # mkdir(F'Case{m}/')
        with open(F'Case{m}/PoblacionInicial{Tamaño}_{M_max[0]}.txt', "wb") as file:
            pickle.dump(P_inicial, file)