# Sistema de Reconhecimento Facial

Um sistema completo de reconhecimento facial que inclui captura de imagens, treinamento de modelo, avaliação e reconhecimento em tempo real.

## Componentes do Sistema

O sistema é composto por quatro módulos principais:

### 1. Detector de Faces (`face_detector.py`)

Ferramenta para capturar e salvar imagens de faces para criar um dataset de treinamento.

- Utiliza InsightFace para detecção precisa de faces
- Mostra pontos de referência faciais (landmarks)
- Opções para salvar automaticamente ou manualmente as faces detectadas
- Redimensiona e prepara as imagens no formato adequado (92x112 pixels em escala de cinza)

### 2. Treinamento do Modelo (`train_model.py`)

Módulo para treinar o modelo de reconhecimento facial a partir do dataset criado.

- Detecta automaticamente as classes (pessoas) com base nas pastas do dataset
- Utiliza todas as imagens exceto as 3 últimas de cada classe para treinamento
- Implementa uma rede neural MLP para reconhecimento
- Salva o modelo treinado com informações das classes para uso posterior

### 3. Avaliador do Modelo (`predictor.py`)

Ferramenta para avaliar o desempenho do modelo treinado.

- Avalia o modelo usando as 3 últimas imagens de cada classe
- Fornece estatísticas detalhadas de precisão por classe
- Permite testar imagens específicas individualmente
- Exibe níveis de confiança para cada predição

### 4. Reconhecimento em Tempo Real (`real_time_recognition.py`)

Sistema para reconhecimento facial em tempo real usando a webcam.

- Detecta e reconhece faces em tempo real
- Verifica se a pessoa está olhando para a câmera
- Exibe landmarks faciais para visualização
- Interface com atalhos de teclado para controlar diferentes recursos

## Requisitos

- Python 3.6+
- PyTorch
- OpenCV
- InsightFace
- NumPy
- PIL (Pillow)

## Instalação

1. Clone este repositório:
```bash
git clone <URL_DO_REPOSITÓRIO>
cd <NOME_DO_REPOSITÓRIO>
```

2. Instale as dependências necessárias:
```bash
pip install torch torchvision opencv-python insightface onnxruntime pillow numpy tqdm
```

## Estrutura do Dataset

O dataset deve seguir a seguinte estrutura:
```
dataset/
  ├── pessoa1/
  │     ├── imagem1.pgm
  │     ├── imagem2.pgm
  │     └── ...
  ├── pessoa2/
  │     ├── imagem1.pgm
  │     └── ...
  └── ...
```

Cada pasta representa uma pessoa/classe e deve conter várias imagens faciais. As 3 últimas imagens (em ordem alfabética) serão reservadas para teste.

## Como Usar

### 1. Criar um Dataset

Use o detector de faces para criar um dataset:
```bash
python face_detector.py
```
- Pressione `s` para salvar faces detectadas
- Pressione `a` para ativar/desativar o salvamento automático
- Pressione `l` para ativar/desativar a exibição de landmarks
- Organize manualmente as imagens salvas em pastas por pessoa

### 2. Treinar o Modelo

Treine o modelo com seu dataset:
```bash
python train_model.py
```
O modelo treinado será salvo como `face_mlp.pth`.

### 3. Avaliar o Modelo

Avalie o desempenho do modelo:
```bash
python predictor.py
```
Escolha a opção 1 para avaliar em todas as imagens de teste ou a opção 2 para testar uma imagem específica.

### 4. Reconhecimento em Tempo Real

Execute o reconhecimento facial em tempo real:
```bash
python real_time_recognition.py
```
- Pressione `l` para ativar/desativar a exibição de landmarks
- Pressione `c` para ativar/desativar a verificação de olhar para a câmera
- Pressione `+`/`-` para ajustar o limite de detecção do olhar
- Pressione `q` para sair

## Personalização

- Ajuste o limite de confiança (`confidence_threshold`) no arquivo `real_time_recognition.py` para controlar a sensibilidade do reconhecimento
- Modifique parâmetros de treinamento como número de épocas, tamanho do batch e taxa de aprendizado no arquivo `train_model.py`

## Notas

- O sistema funciona melhor com boa iluminação e quando a pessoa está olhando diretamente para a câmera
- Para melhor precisão, inclua pelo menos 5-10 imagens por pessoa no dataset de treinamento
- As imagens são armazenadas no formato PGM e redimensionadas para 92x112 pixels, seguindo o padrão do AT&T Database of Faces
