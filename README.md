# Sistema de Reconhecimento Facial

Um sistema completo de reconhecimento facial que inclui captura de imagens, treinamento de modelo, avaliação e reconhecimento em tempo real.

## Componentes do Sistema

O sistema é composto por cinco módulos principais:

### 1. Download de Modelos (`download_models.py`)

Ferramenta para baixar e configurar os modelos necessários para o sistema.

- Verifica a versão do Python (3.7+)
- Instala automaticamente as dependências necessárias
- Baixa os modelos do InsightFace para detecção facial e landmarks
- Cria a estrutura de diretórios necessária

### 2. Detector de Faces (`face_detector.py`)

Ferramenta para capturar e salvar imagens de faces para criar um dataset de treinamento.

- Utiliza InsightFace para detecção precisa de faces
- Mostra pontos de referência faciais (landmarks)
- Opções para salvar automaticamente ou manualmente as faces detectadas
- Aplica pré-processamento avançado às imagens:
  - Alinhamento facial usando pontos de referência
  - Equalização de histograma para melhorar o contraste
  - Realce de bordas para destacar características faciais
- Redimensiona as imagens para 92x112 pixels em escala de cinza

### 3. Treinamento do Modelo (`train_model.py`)

Módulo para treinar o modelo de reconhecimento facial a partir do dataset criado.

- Detecta automaticamente as classes (pessoas) com base nas pastas do dataset
- Utiliza todas as imagens exceto as 3 últimas de cada classe para treinamento
- Aplica o mesmo pré-processamento usado na captura de imagens
- Implementa uma rede neural MLP para reconhecimento
- Salva o modelo treinado com informações das classes para uso posterior

### 4. Avaliador do Modelo (`predictor.py`)

Ferramenta para avaliar o desempenho do modelo treinado.

- Avalia o modelo usando as 3 últimas imagens de cada classe
- Fornece estatísticas detalhadas de precisão por classe
- Permite testar imagens específicas individualmente
- Exibe níveis de confiança para cada predição

### 5. Reconhecimento em Tempo Real (`real_time_recognition.py`)

Sistema para reconhecimento facial em tempo real usando a webcam.

- Detecta e reconhece faces em tempo real
- Aplica o mesmo pré-processamento usado no treinamento
- Verifica se a pessoa está olhando para a câmera
- Exibe landmarks faciais para visualização
- Interface com atalhos de teclado para controlar diferentes recursos

## Requisitos

- Python 3.7+
- PyTorch
- OpenCV
- InsightFace
- NumPy
- PIL (Pillow)
- ONNX Runtime

## Instalação

1. Clone este repositório:
```bash
git clone <URL_DO_REPOSITÓRIO>
cd <NOME_DO_REPOSITÓRIO>
```

2. Execute o script de download de modelos:
```bash
python download_models.py
```

Este script irá:
- Verificar a versão do Python
- Instalar todas as dependências necessárias
- Baixar os modelos do InsightFace
- Configurar a estrutura de diretórios

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

### 1. Configuração Inicial

Execute o script de download de modelos:
```bash
python download_models.py
```

### 2. Criar um Dataset

Use o detector de faces para criar um dataset:
```bash
python face_detector.py
```
- Pressione `s` para salvar faces detectadas
- Pressione `a` para ativar/desativar o salvamento automático
- Pressione `l` para ativar/desativar a exibição de landmarks
- Organize manualmente as imagens salvas em pastas por pessoa

### 3. Treinar o Modelo

Treine o modelo com seu dataset:
```bash
python train_model.py
```
O modelo treinado será salvo como `face_mlp.pth`.

### 4. Avaliar o Modelo

Avalie o desempenho do modelo:
```bash
python predictor.py
```
Escolha a opção 1 para avaliar em todas as imagens de teste ou a opção 2 para testar uma imagem específica.

### 5. Reconhecimento em Tempo Real

Execute o reconhecimento facial em tempo real:
```bash
python real_time_recognition.py
```
- Pressione `l` para ativar/desativar a exibição de landmarks
- Pressione `c` para ativar/desativar a verificação de olhar para a câmera
- Pressione `+`/`-` para ajustar o limite de detecção do olhar
- Pressione `q` para sair

## Pré-processamento de Imagens

O sistema aplica um pré-processamento avançado a todas as imagens faciais:

1. **Alinhamento Facial**: Usa os pontos de referência dos olhos para alinhar o rosto horizontalmente
2. **Equalização de Histograma**: Normaliza o contraste da imagem para lidar com variações de iluminação
3. **Realce de Bordas**: Aplica operadores Sobel para destacar características faciais importantes

Este pré-processamento consistente entre captura, treinamento e reconhecimento em tempo real melhora significativamente a precisão do sistema.

## Personalização

- Ajuste o limite de confiança (`confidence_threshold`) no arquivo `real_time_recognition.py` para controlar a sensibilidade do reconhecimento
- Modifique parâmetros de treinamento como número de épocas, tamanho do batch e taxa de aprendizado no arquivo `train_model.py`
- Ajuste o fator de realce de bordas (`alpha`) nos arquivos de pré-processamento para controlar a intensidade do realce

## Notas

- O sistema funciona melhor com boa iluminação e quando a pessoa está olhando diretamente para a câmera
- Para melhor precisão, inclua pelo menos 5-10 imagens por pessoa no dataset de treinamento
- As imagens são armazenadas no formato PGM e redimensionadas para 92x112 pixels, seguindo o padrão do AT&T Database of Faces
- O pré-processamento avançado ajuda a lidar com variações de iluminação e orientação facial
