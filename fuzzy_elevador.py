import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Definindo as variáveis fuzzy
peso_elevador = ctrl.Antecedent(np.arange(1000, 1501, 1), 'peso_elevador')
andar_chamado = ctrl.Antecedent(np.arange(0, 11, 1), 'andar_chamado')
num_andares = ctrl.Antecedent(np.arange(1, 11, 1), 'num_andares')

# Variável de saída: Condição de operação do elevador
condicao = ctrl.Consequent(np.arange(0, 101, 1), 'condicao')

# Definindo as funções de pertinência para cada variável
peso_elevador['leve'] = fuzz.trapmf(peso_elevador.universe, [1000, 1000, 1250, 1500])
peso_elevador['medio'] = fuzz.trimf(peso_elevador.universe, [1250, 1375, 1500])
peso_elevador['pesado'] = fuzz.trapmf(peso_elevador.universe, [1000, 1250, 1250, 1500])

andar_chamado['baixo'] = fuzz.trimf(andar_chamado.universe, [0, 0, 50])
andar_chamado['medio'] = fuzz.trimf(andar_chamado.universe, [25, 50, 75])
andar_chamado['alto'] = fuzz.trimf(andar_chamado.universe, [50, 100, 100])

num_andares['pequeno'] = fuzz.trimf(num_andares.universe, [1, 1, 50])
num_andares['medio'] = fuzz.trimf(num_andares.universe, [25, 50, 75])
num_andares['alto'] = fuzz.trimf(num_andares.universe, [50, 100, 100])

condicao['eficiente'] = fuzz.trapmf(condicao.universe, [0, 0, 25, 50])
condicao['moderada'] = fuzz.trimf(condicao.universe, [25, 50, 75])
condicao['ineficiente'] = fuzz.trapmf(condicao.universe, [50, 75, 100, 100])

# Definindo as regras fuzzy
rule1 = ctrl.Rule(peso_elevador['leve'] & andar_chamado['baixo'] & num_andares['pequeno'], condicao['eficiente'])
rule2 = ctrl.Rule(peso_elevador['pesado'] & andar_chamado['alto'] & num_andares['alto'], condicao['ineficiente'])
rule3 = ctrl.Rule(peso_elevador['medio'] & andar_chamado['medio'] & num_andares['medio'], condicao['moderada'])
rule4 = ctrl.Rule(peso_elevador['leve'] & andar_chamado['alto'] & num_andares['pequeno'], condicao['moderada'])
rule5 = ctrl.Rule(peso_elevador['pesado'] & andar_chamado['baixo'] & num_andares['alto'], condicao['ineficiente'])
rule6 = ctrl.Rule(~(peso_elevador['leve'] | peso_elevador['medio'] | peso_elevador['pesado']) |
                 ~(andar_chamado['baixo'] | andar_chamado['medio'] | andar_chamado['alto']) |
                 ~(num_andares['pequeno'] | num_andares['medio'] | num_andares['alto']), condicao['moderada'])

# Criando o sistema de controle fuzzy
condicao_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
condicao_simulador = ctrl.ControlSystemSimulation(condicao_ctrl)

# Função para simular a condição do elevador
def simular_condicao(peso, andar, num_andares):
    condicao_simulador.input['peso_elevador'] = peso
    condicao_simulador.input['andar_chamado'] = andar
    condicao_simulador.input['num_andares'] = num_andares
    
    condicao_simulador.compute()
    
    return condicao_simulador.output['condicao']

# Exemplo de uso
peso = 1200  # Peso do elevador, incluindo passageiros
andar = 20   # Andar em que o elevador está sendo chamado
andares = 30  # Número total de andares do prédio

resultado = simular_condicao(peso, andar, andares)
print(f"Condição de operação do elevador: {resultado:.2f}")

# Visualizando as funções de pertinência
peso_elevador.view()
andar_chamado.view()
num_andares.view()
condicao.view()

# Superfície de decisão
def plot_decision_surface():
    x = np.arange(1000, 1501, 10)
    y = np.arange(0, 101, 10)
    z = np.zeros((len(x), len(y)))
    
    for i in range(len(x)):
        for j in range(len(y)):
            z[i, j] = simular_condicao(x[i], y[j], 50)  # Usando 50 como número fixo de andares
    
    X, Y = np.meshgrid(x, y)
    Z = z.T
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    
    ax.set_xlabel('Peso do Elevador (kg)')
    ax.set_ylabel('Andar Chamado')
    ax.set_zlabel('Condição de Operação (%)')
    
    plt.show()

plot_decision_surface()
