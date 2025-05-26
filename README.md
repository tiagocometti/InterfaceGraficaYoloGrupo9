# Detector de Placas - Interface Gráfica com YOLOv9

## Sobre o Projeto

- Trabalho desenvolvido pelo **Grupo 9** para a disciplina de **Inteligência Artificial**
- Alunos: Tiago Cometti, Lucas Leão, João Marcos Maia e Jean Carlos Vieira

## Como Executar o Projeto

### 1. Clone o repositório:

```bash
git clone https://github.com/tiagocometti/InterfaceGraficaYoloGrupo9.git
cd InterfaceGraficaYoloGrupo9
```

### 2. Crie um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
venv\Scripts\activate  # No Windows
# ou
source venv/bin/activate  # No Linux/Mac
```

### 3. Instale as dependências:

```bash
pip install -r requirements.txt
```

### 4. Execute o programa:

```bash
python interface_grafica.py
```

## Estrutura da Pasta

```bash
InterfaceYolo/
│
├── yolov9/                 # Arquitetura YOLOv9
├── pesos/                  # Pesos do modelo (.pt)
│   └── best.pt             # Modelo treinado
├── interface_grafica.py    # Código principal da interface
├── requirements.txt        # Dependências
└── README.md               # Instruções
```

## Trocar os Pesos do Modelo

Se quiser utilizar outro modelo treinado, basta:

Substituir o arquivo localizado em:

```bash
/pesos/best.pt
```

Por outro arquivo .pt de sua preferência.

## Observações Importantes

- A pasta `yolov9/` precisa estar sempre na mesma pasta que o arquivo `interface_grafica.py`.
- O arquivo de pesos `best.pt` deve obrigatoriamente estar dentro da pasta `pesos/`.
- O Tkinter já vem instalado por padrão no Python, portanto não é necessário instalar manualmente.
- Caso queira alterar o modelo, basta substituir o arquivo na pasta `/pesos/`.
