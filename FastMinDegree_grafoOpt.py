######### codigo 1 ########################

import numpy as np
from copy import deepcopy
from glib import gphs
# import networkx as nx
import time
import pickle
import os

class grafo:
    """
    Construir grafo a partir de su matriz de adyacencia, lista de adyacencia
    o lista de aristas.
    """
    def __init__(self, name: str, tamaño: int, matriz: list = [], lista: list = [], 
                 edges: list[int] = [], ML: str = 'matrizAdy', n_degree: int = 60):
        match ML:
            case 'matrizAdy':      
            ### Construir grafo a partir de una matriz de adyacencia
            
                self.matrizAdy = matriz
                self.tamaño = len(matriz)
                # creamos su lista de adyacencia correspondiente
                self.listaAdy = []
                # creamos la lista de sus aristas: las aristas se empiezan a contar desde 1, y 
                #           corresponde a la entrada i-1 de la lista de adyacencia y matriz
                self.edges = np.array([[], []]).T
                for i in range(self.tamaño):
                    self.listaAdy.append([True, np.array([], dtype=int)])
                    for j in range(self.tamaño):
                        if matriz[i, j]:
                            self.listaAdy[i][1] = np.append(self.listaAdy[i][1], j)
                            # Guardamos ambas representaciones de la arista
                            self.edges = np.append(self.edges, [[i+1, j+1]], axis=0) 
                            
            case 'listaAdy':       
            ### Construir nodo a partir de una lista de adyacencia
            
                self.listaAdy = []
                self.tamaño = len(lista)
                self.matrizAdy = np.zeros((self.tamaño, self.tamaño), dtype=int)
                # creamos la lista de sus aristas: las aristas se empiezan a contar desde 1, y 
                #           corresponde a la entrada i-1 de la lista de adyacencia y matriz
                self.edges = np.array([[], []]).T
                for i in range(self.tamaño):
                    # print(i, lista[i])
                    self.listaAdy.append([True, lista[i]])
                    for j in lista[i]:
                        if not self.matrizAdy[i, j-1]:
                            self.matrizAdy[i, j-1] = 1
                            self.matrizAdy[j-1, i] = 1
                        # Guardamos ambas representaciones de la arista
                        self.edges = np.append(self.edges, [[i+1, j]], axis=0) 
                    # print(self.matrizAdy)
        
            case 'listaEdges':       
            ### Construir nodo a partir de una lista de aristas
            
                #guardamos la lista de aristas
                self.edges = np.array(deepcopy(edges))
                # Guardamos el numero de nodos en el grafo como tamaño
                self.tamaño = self.edges.max()
                
                # creamos la matriz de adyacencia
                self.matrizAdy = np.zeros((self.tamaño, self.tamaño), dtype=int)
                # Llenamos la mitad de la matriz de adyacencia (la triangular superior)
                for [i,j] in self.edges:
                    if not self.matrizAdy[i-1,j-1] and i > j:
                        self.matrizAdy[i-1,j-1] = 1
                self.matrizAdy = self.matrizAdy + self.matrizAdy.T
                # Usaremos una lista auxiliar con las entradas de la matriz (indices) 
                # para crear la lista de adyacencia
                indices = np.where(self.matrizAdy==1)
                # print(indices)
                self.listaAdy = []
                for i in range(self.tamaño):
                    # Los nodos activos tienen en la primera entrada True, los inactivos: False
                    # la segunda entrada es la lista de nodos adyacentes
                    # Recordemos que los nodos inician en 1
                    self.listaAdy.append([True, np.array([], dtype=int)])
                    j1 = indices[1][np.where(indices[0]==i)[0]]
                    j2 = indices[0][np.where(indices[1]==i)[0]]
                    # print(i,j1,j2)
                    # Si la lista de aristas tiene tanto ambas aristas: [i,j] y [j,i]
                    # solo es necesario revisar j1
                    for j in j1:
                        if j+1 not in self.listaAdy[i][1]:
                            self.listaAdy[i][1] = np.append(self.listaAdy[i][1], j+1)
                    # Si solo tiene una representación de la arista es necesario revisar también j2
                    for j in j2:
                        if j+1 not in self.listaAdy[i][1]:
                            self.listaAdy[i][1] = np.append(self.listaAdy[i][1], j+1)
                    # if i < 100:
                    #     print(i+1,self.listaAdy[i][1])
            
            # self.matrizAdy = self.matrizAdy + self.matrizAdy.T
            
        # grados: característica del grafo donde se guarda el grado de todos sus
        #           sus nodos en una lista, donde la entrada corresponde al numero de nodos  
        #           iniciando en cero
        self.grados = []
        # Calculamos los gados de cada nodo
        self.calculaGrados()
        # Guardamos la cantidad de aristas como len_edges
        self.len_edges = int(self.matrizAdy.sum()/2)
        # Guardamos la cantidad de nodos que hay de cada grado desde cero hasta n_degree
        # Por default n_degree = 60
        self.N_grados = np.zeros(n_degree)
        self.CuantosNgradok(n_degree)
        self.name = name
        
    def CuantosNgradok(self, n_degree: int) -> None: 
        """
        Regresa una lista de tamaño k, 
        donde k = es el grado máximo que nos interesa,
        y la entrada i corresponde a la cantidad de nodos de grado i
        """
        for i in range(n_degree):
            self.N_grados[i] = np.sum(self.grados == i)

    def calculaGrados(self) -> None:
        """
        Calculamos el grado de un nodo usando la lista de adyacencia.
        Si el nodo es inactivo (entrada False), su grado es cero.
        """
        self.grados = np.array([len(self.listaAdy[i][1])*self.listaAdy[i][0] for i in range(self.tamaño)])

    def EdgestoMatriz(self) -> None:        
        """
        Llenamos las entradas faltantes en la matriz 
        a partir de la lista de aristas
        """
        for [i, j] in self.edges:
            if not self.matrizAdy[i-1, j-1] and i > j:
                self.matrizAdy[i-1, j-1] = 1
                self.matrizAdy[j-1, i-1] = 1

    def NodoGradMin(self) -> int|bool:
        # print(self.grados)
        if len(self.grados[self.grados > 0]):
            m = np.where(self.grados == min(self.grados[self.grados > 0]))[0][0]
            return m
        else: 
            return False
    
    def Extension(self, Eliminacion: str = 'FMD', orden: list[int] = [], 
                  ver: bool = False, ver2: bool = False, ver3: bool = False) -> tuple:
        """
        Esta función regresa una extensión cordal del grafo dado
        Si Eliminacion = 'FMD', se sigue el algoritmo de Fast Minimo Degree
        Si Eliminacion = 'Orden', se realiza el mismo proceso del algoritmo, 
                              pero no se considera el nodo de menor grado,
                              sino el orden de eliminación de nodos dado
        """
        t = time.time()
        # Creamos un grafo auxiliar donde haremos eliminación de nodos,
        # uniremos los nodos adyacentes al nodo eliminado 
        # y agregaremos las aristas correspondientes
        G_aux = deepcopy(self)
        # Creamos el grafo donde se guardará la estensión cordal
        # en este solo se agregaran las aristas agregadas en el grafo auxiliar
        G_ext = deepcopy(self)
        m = self.tamaño
        name = self.name
        if ver2 or ver:
            # Mostramos y guardamos el proceso de hacer la extensión con el tiempo tanscurrido
            # Cada 100 nodos
            A0 = self.len_edges
            
            if not os.path.isdir(F'Case{m}/'):
                os.mkdir(F'Case{m}/')
                
            i_f = True
            j_f = 0
            while(i_f):
                if os.path.isdir(F'Case{m}/ExtensionTOpt{m}_{j_f}'):
                    j_f += 1
                else:
                    i_f = False
                
            if Eliminacion == 'FMD':
                with open(F"Case{m}/ExtensionTOpt{m}_{j_f}.txt","w") as file: 
                    file.write('Grafo: ' + name + '\n')
                
                f = open(F'Case{m}/ExtensionTOpt{m}_{j_f}.txt', 'a')
                
            if Eliminacion == 'Orden':
                with open(F"Case{m}/ExtensionTOpt{m}_{j_f}.txt","w") as file: 
                    file.write('Grafo: ' + name + '\n')
                
                f = open(F'Case{m}/ExtensionTOpt{m}_{j_f}.txt', 'a')
                
            W = F'\n Aristas del grafo original: {A0}'
            # print(W)
            f.write(W + '\n')
            W = '\n   Extensión del grafo obtenida con un orden de eliminación de nodos dado:'
            W += '\n A: Aristas agregadas'
            W += '\n (Cantidad de nodos eliminados); Aristas totales: ( ) -> tiempo transcurrido (HH:MM:Segundos)'
            
            # print(W)
            f.write(W + '\n')
            
        for l in range(self.tamaño-1):
            # print('l', l)
            if Eliminacion == 'FMD':  #Fast Minimo Degree
                if l == 0:
                    orden = np.zeros(self.tamaño, dtype=int)
                # En cada ocasión calcularemos los grados de los nodos
                G_aux.calculaGrados()
                ed_j = G_aux.NodoGradMin()
                orden[l] = ed_j
                
            if Eliminacion == 'Orden':    # Se da el orden de eliminación manualmente
                ed_j = orden[l]
            
            if ver:
                G_aux.calculaGrados()
                W += F'\n nodo: {ed_j+1}, grado: {G_aux.grados[ed_j]}'
                
            # Desactivamos el nodo eliminado
            # print(ed_j)
            G_aux.listaAdy[ed_j][0] = False
            # Creamos una lista auxiliar con los nodos adyacentes del nodo eliminado
            aux = np.array(G_aux.listaAdy[ed_j][1])
            if ver:
                W += F'\n  Nodos adyacentes: {aux}'
                W += '\n  Nuevas aristas:'
            n = len(aux)
            
            # Recorremos cada nodo adyacente
            for i1 in range(n):
                if ver3:
                    W += '\n'
                    W += F' {i1},'
                    W += F' {aux},'
                    W += F'\n Lista Adyacencia aux: {G_aux.listaAdy[aux[i1]-1]}'
                # Eliminamos el nodo eliminado de la lista adyacente de los nodos en aux
                G_aux.listaAdy[aux[i1]-1][1] = np.delete(G_aux.listaAdy[aux[i1]-1][1], G_aux.listaAdy[aux[i1]-1][1] == ed_j +1)
                
                # Revisamos si las aristas ya están en el grafo, para eso usamos un vector y
                # una matriz auxiliares
                vector = np.reshape(G_aux.listaAdy[aux[i1]-1][1], (len(G_aux.listaAdy[aux[i1]-1][1]),1))
                matriz = np.repeat(vector, n, axis=1)
                
                #Obtenemos los índices que indican las aristas que ya estan en el grafo
                _, i2 = np.where(matriz == aux)
                
                # Quitamos los indices de las aristas repetidas
                unir = np.delete(aux, i2)
                if ver3:
                    W += '\n'
                    W += F'\n {i2}'
                    W += F'\n Unir: {unir}'
                unir = np.delete(unir, unir == aux[i1])
                if ver3:
                    W += F'\n {aux[i1]}'
                    W += F'\n Unir: {unir}'
                    
                # Si hay nuevas aristas a agregar
                if len(unir) > 0:
                    #Actualizamos las entradas de la matriz adyacente de la extensión 
                    G_ext.matrizAdy[aux[i1]-1][unir-1] = 1
                    
                    # Actualizamos las listas adyacentes de la extensión y del grafo auxiliar
                    G_ext.listaAdy[aux[i1]-1][1] = np.append(G_ext.listaAdy[aux[i1]-1][1], unir)
                    G_aux.listaAdy[aux[i1]-1][1] = np.append(G_aux.listaAdy[aux[i1]-1][1], unir)
                    for j in unir:
                        #Agregamos sólo las nuevas aristas que no están en el grafo original
                        G_ext.edges = np.append(G_ext.edges, [[aux[i1], j-1], [j-1, aux[i1]]], axis=0)
                        #Actualizamos las listas adyacentes de los nodos adyacentes
                        G_ext.listaAdy[j-1][1] = np.append(G_ext.listaAdy[j-1][1], aux[i1])
                        G_aux.listaAdy[j-1][1] = np.append(G_aux.listaAdy[j-1][1], aux[i1])
                        G_ext.matrizAdy[j-1][aux[i1]-1] = 1
                        if ver:
                            W += F'\n{[aux[i1], j-1], [j-1, aux[i1]]}'
                if ver3:
                    W += r'\n'
                    W += F'\n Lista Adyacencia aux: {G_aux.listaAdy[aux[i1]-1]}'
            if ver:
                W += r'\n'
                W += F'\n  Aristas: {G_ext.edges.tolist()}'
                W += F'\n  Cantidad de aristas: {int(G_ext.matrizAdy.sum()/2)}'
                W += r'\n'
            if ver2:
                if (l+1) % 100 == 0:
                    A = int(G_ext.matrizAdy.sum()/2)
                    W += F'\n A: {A-A0}'
                    A0 = A
                    W += F'\n {l+1}; Aristas: {int(G_ext.matrizAdy.sum()/2)} -> {segundos_a_segundos_minutos_y_horas(time.time() - t)}'
                    # print(W)
                    f.write(W + '\n')
        # Calculamos la cantidad de aristas en la extensión
        G_ext.len_edges = int(G_ext.matrizAdy.sum()/2)
        # Calculamos los nuevos grados de cada nodo en la extensión y la lista de N_grados
        G_ext.calculaGrados()
        for i in range(len(G_ext.N_grados)):
            G_ext.N_grados[i] = np.sum(G_ext.grados == i)
        if ver2:
            W += F'\n Tiempo en crear la extensión: {segundos_a_segundos_minutos_y_horas(time.time() - t)}'
            W += F'\n Cantidad de aristas: {G_ext.len_edges}'
            # print(W)
            f.write(W)   
            f.close() 
        return G_ext, G_aux, orden
        
def segundos_a_segundos_minutos_y_horas(segundos):
    horas = int(segundos / 60 / 60)
    segundos -= horas*60*60
    minutos = int(segundos/60)
    segundos -= minutos*60
    return f"{horas:02}:{minutos:02}:{segundos:002.10f}"

if __name__ == "__main__":
    m_0 = [500, 300, 118]
    name_0 = ['pglib_opf_case500_goc', 'case300', 'case118']
    G_fit_0 = [1220, 661, 265]
    
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
            os.mkdir(F'Case{m}/')
        with open(F'Case{m}/Grafo.txt', "wb") as file:
            pickle.dump(G, file)
        print(F'Creación de grafo: {segundos_a_segundos_minutos_y_horas(time.time() - t)}')
        
    