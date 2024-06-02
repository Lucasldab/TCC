import numpy as np

# Definição da função de Rastrigin
def rastrigin(X):
    # Calcula o valor da função de Rastrigin para um vetor X
    return 10 * len(X) + sum([x**2 - 10 * np.cos(2 * np.pi * x) for x in X])

# Classe PSO
class PSO:
    def __init__(self, n_particulas, dimensoes, n_iteracoes, limites, w_min=0.1, w_max=0.9, v_max=0.1):
        self.n_particulas = n_particulas  # Define o número de partículas
        self.dimensoes = dimensoes  # Define a quantidade de dimensões do espaço de busca
        self.n_iteracoes = n_iteracoes  # Define o número total de iterações
        self.limites = limites  # Define os limites mínimos e máximos do espaço de busca
        self.w_min = w_min  # Define o fator de inércia mínimo
        self.w_max = w_max  # Define o fator de inércia máximo
        self.v_max = v_max  # Define a velocidade máxima para a atualização da velocidade das partículas

        # Inicializa as posições das partículas de forma aleatória dentro dos limites especificados
        self.posicoes = np.random.rand(n_particulas, dimensoes) * (limites[1] - limites[0]) + limites[0]
        # Inicializa as velocidades das partículas como zero
        self.velocidades = np.zeros((n_particulas, dimensoes))
        # Inicializa as melhores posições pessoais das partículas como suas posições iniciais
        self.pbest_posicoes = self.posicoes.copy()
        # Inicializa os melhores scores pessoais como infinito, pois ainda não foram avaliados
        self.pbest_scores = np.array([float('inf')] * n_particulas)
        # Inicializa o melhor score global como infinito
        self.gbest_score = float('inf')
        # Inicializa a melhor posição global como um vetor de infinitos
        self.gbest_posicao = np.array([float('inf'), float('inf')])

    # Método para atualizar a velocidade das partículas
    def atualiza_velocidade(self, iteracao):
        # Calcula o fator de inércia baseado na iteração atual
        w = self.w_max - (self.w_max - self.w_min) * (iteracao / self.n_iteracoes)
        for i in range(self.n_particulas):
            r1, r2 = np.random.rand(2)
            # Atualiza a velocidade da partícula baseado nas melhores posições pessoais e global
            nova_velocidade = (w * self.velocidades[i] +
                               r1 * (self.pbest_posicoes[i] - self.posicoes[i]) +
                               r2 * (self.gbest_posicao - self.posicoes[i]))
            # Aplica o limite de velocidade
            self.velocidades[i] = np.clip(nova_velocidade, -self.v_max, self.v_max)

    # Método para atualizar a posição das partículas e aplicar o método Damping
    def atualiza_posicao(self):
        for i in range(self.n_particulas):
            # Atualiza a posição da partícula baseado em sua velocidade
            self.posicoes[i] += self.velocidades[i]
            # Para cada dimensão, verifica se a partícula excedeu os limites
            for dim in range(self.dimensoes):
                if self.posicoes[i][dim] < self.limites[0]:
                    # Se menor que o limite mínimo, ajusta para o limite e inverte a velocidade
                    self.posicoes[i][dim] = self.limites[0]
                    self.velocidades[i][dim] *= -1
                elif self.posicoes[i][dim] > self.limites[1]:
                    # Se maior que o limite máximo, ajusta para o limite e inverte a velocidade
                    self.posicoes[i][dim] = self.limites[1]
                    self.velocidades[i][dim] *= -1

    # Método para avaliar a função objetivo nas posições das partículas
    def avalia(self):
        for i in range(self.n_particulas):
            # Calcula o score (valor da função objetivo) da partícula
            score = rastrigin(self.posicoes[i])
            # Se o score atual é melhor que o melhor pessoal, atualiza o melhor pessoal
            if score < self.pbest_scores[i]:
                self.pbest_scores[i] = score
                self.pbest_posicoes[i] = self.posicoes[i].copy()
            # Se o score atual é melhor que o melhor global, atualiza o melhor global
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest_posicao = self.posicoes[i].copy()

    # Método para executar o algoritmo PSO
    def executa(self):
        for iteracao in range(self.n_iteracoes):
            self.atualiza_velocidade(iteracao)
            self.atualiza_posicao()
            self.avalia()
            print(f"Iteração {iteracao+1}/{self.n_iteracoes}, Melhor Score Global: {self.gbest_score}")
        return self.gbest_posicao, self.gbest_score

# Parâmetros para a execução do PSO
n_particulas = 30
dimensoes = 2
n_iteracoes = 100
limites = (-5.12, 5.12)
v_max = 0.2  # Limite máximo para a velocidade das partículas

# Executa o PSO
pso = PSO(n_particulas, dimensoes, n_iteracoes, limites, v_max=v_max)
melhor_posicao, melhor_score = pso.executa()
print(f"Melhor posição: {melhor_posicao}, Melhor score: {melhor_score}")
