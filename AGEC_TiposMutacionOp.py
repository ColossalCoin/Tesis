######### codigo 2 ########################

import numpy as np
from FastMinDegree_grafoOpt import grafo
from glib import gphs
import networkx as nx
from copy import copy, deepcopy

class Orden:
    def __init__(self, inicial = [], orden = []):
        self.inicial = inicial
        self.orden = orden
        self.tamaño = len(inicial)
        
    def SwapMutationCrea(self):
        a = np.random.choice(self.inicial)
        B = [i for i in range(a, self.tamaño)]
        b = np.random.choice((B-a) < 4,)
        print(a,b)
        self.orden[a], self.orden[b] = self.orden[b], self.orden[a]
        return a, b
    
    def SwapMutation(self):
        a, b = np.random.choice(self.inicial, size=(2), replace=False)
        self.orden[a], self.orden[b] = self.orden[b], self.orden[a]
        
    def InsertMutation(self):
        a, b = np.random.choice(self.tamaño, size=(2), replace=False)
        m, M = min(a,b), max(a,b)
        self.orden[m+1], self.orden[m+2: M+1] = self.orden[M], self.orden[m+1: M]
    
    def ScrambleMutation(self):
        a, b = np.random.choice(self.tamaño, size=(2), replace=False)
        m, M = min(a,b), max(a,b)
        new = deepcopy(self.orden[m:M+1])
        np.random.shuffle(new)
        self.orden[m: M+1] = new
    
    def InversionMutation(self):
        a, b = np.random.choice(self.tamaño, size=(2), replace=False)
        m, M = min(a,b), max(a,b) 
        if m:
            new = deepcopy(self.orden[M:m-1:-1])
        else:
            new = deepcopy(self.orden[M::-1])
        self.orden[m: M+1] = new
    
    def PMX(self, otro):
        a, b = np.random.choice(self.tamaño, size=(2), replace=False)
        m, M = min(a,b), max(a,b)
        hijo = deepcopy(self)
        off = [i for i in range(m, M+1)]
        for i in range(m, M+1):
            if otro.orden[i] not in self.orden[m: M+1]:
                k = i
                p = 0
                while(p < self.tamaño):
                    if self.orden[k] not in otro.orden[m: M+1]:
                        J = np.where(otro.orden == self.orden[k])
                        if len(J[0]):
                            j = J[0][0]
                            hijo.orden[j] = otro.orden[i]
                            off.append(j)
                        break
                    else:
                        k = np.where(otro.orden == self.orden[k])[0][0]
                    p += 1
        for i in range(self.tamaño):
            if i not in off:
                hijo.orden[i] = otro.orden[i]
        return hijo
                

    
    
    
    
    
    
    
    
    
    