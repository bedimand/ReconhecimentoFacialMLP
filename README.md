# Sistema de Reconhecimento Facial com MLP

Este projeto implementa um sistema completo de reconhecimento facial baseado em uma Rede Neural Perceptron Multicamadas (MLP), contemplando etapas de coleta, pré-processamento, treinamento, avaliação e inferência em tempo real.

---

## 📂 Estrutura do Projeto

- **main.py**            : Ponto de entrada. Exibe menu interativo para executar cada etapa do fluxo.
- **config.yaml**        : Arquivo de configuração com parâmetros (caminhos, tamanho de imagem, hiperparâmetros, controles).
- **src/**               : Código-fonte principal organizado por funcionalidade:
  - **model/**              : Arquivos relacionados ao modelo neural:
    - **model.py**          : Definição da classe `FaceRecognitionMLP`, funções de salvar e carregar modelo.
    - **train.py**          : Função `train_model` para treinamento com balanceamento de classes e manifest.
    - **evaluation.py**     : Funções de avaliação (acurácia, relatório de classificação e matriz de confusão).
  - **face/**               : Processamento de faces:
    - **face_detection.py** : Detecção via InsightFace e extração de crops centralizados no nariz.
    - **face_recognition.py** : `TransformFactory` para transformações e rotina de inferência (`recognize_face`).
    - **preprocessing.py**  : Módulo de remoção de fundo (MediaPipe) e normalização.
  - **data/**               : Manipulação de dados:
    - **datasets.py**       : Definições de conjuntos de dados e obtenção de classes.
    - **preprocessing_dataset.py** : Rotina em lote para pré-processar dataset completo.
  - **utils/**              : Utilitários diversos:
    - **utils.py**          : Utilitários gerais (console, contagem de imagens, criação de pastas).
    - **visualization.py**  : Desenho de bounding boxes, landmarks, estatísticas e previews.
    - **config.py**         : Carrega `config.yaml` e fornece métodos de acesso às configurações.
    - **capture_faces.py**  : Script para extrair faces de uma pasta de imagens (modo batch).
  - **frame_processor.py**  : Integra detecção e reconhecimento em um frame de vídeo.

---

## 📝 Requisitos

- Python 3.7 ou superior
- Pacotes (via `pip install -r requirements.txt`):
  - opencv-python
  - torch, torchvision
  - insightface, onnxruntime
  - mediapipe
  - pyyaml
  - numpy, Pillow
  - scikit-learn, matplotlib, tqdm

- (Opcional) GPU compatível com CUDA para aceleração

---

## ⚙️ Configuração

1. Renomeie e adapte `config.yaml` conforme seu ambiente:
   - `paths.dataset_dir`   : Diretório para armazenar imagens coletadas
   - `paths.model_save_path`: Caminho para salvar o peso do modelo (`.pth`)
   - `image_processing.target_size`: Tamanho (L×A) do crop de face
   - Hiperparâmetros de treinamento (`training.*`)
   - Controles de teclado e cores de visualização

2. (Opcional) Baixe os modelos do InsightFace em `models/insightface` ou ajuste o `root` no código.

---

## 🚀 Fluxo de Uso

Execute:

```bash
python main.py
```

O menu oferece as opções:

1. **Detectar e coletar faces (webcam)**
   - Faz detecção em tempo real, desenha landmarks e captura crops centralizados no nariz.
   - Aperte tecla configurada para salvar faces individuais ou ative auto-save.

2. **Capturar faces de pasta de imagens**
   - Processa todas as imagens de uma pasta, com ou sem visualização, salvando crops no dataset.

3. **Treinar modelo de reconhecimento**
   - Pré-requisito: execute pré-processamento ou tenha `_preprocessed` no dataset.
   - Gera um manifest para separar treino e validação, treina o MLP e salva pesos e classes.

4. **Avaliar modelo de reconhecimento**
   - Carrega o modelo salvo, exclui imagens de treino via manifest e calcula acurácia e matriz de confusão.

5. **Reconhecimento em tempo real**
   - Captura vídeo da webcam, detecta faces, aplica remoção de fundo, reconhece usando o MLP e exibe rótulos.

6. **Pré-processar e exportar imagem única**
   - Permite carregar e pré-processar um arquivo de imagem, salvando o resultado normalizado.

7. **Pré-processar todas as imagens**
   - Executa em lote o pipeline de remoção de fundo + resize em todo o dataset.

8. **Sair**

---

## 🔍 Detalhes Técnicos

### Detecção e Extração
- InsightFace (`FaceAnalysis`) detecta bounding boxes, 106 landmarks e 5 pontos-chave (kps).
- `extract_face` centraliza o crop no nariz, dimensiona baseado na distância interpupilar e redimensiona ao tamanho configurado.

### Pré-processamento
- Offline: `PreprocessModule` usa MediaPipe para remover o fundo antes de salvar imagens em `_preprocessed`.
- Em tempo real: aplica `remove_background_grabcut` antes de normalizar e enviar ao modelo.
- Normalização: pipeline `ToTensor` + `Normalize` (médias e desvios padrão de ImageNet ou configurados).

### Arquitetura do Modelo
- MLP com camadas fully-connected:  input→2048→1024→512→128→num_classes.
- Dropout opcional, Adam optimizer, CrossEntropyLoss, scheduler e early stopping configuráveis.
- Balanceamento de classes via amostragem de instâncias e manifest.

#### 🧠 Entendendo o MLP de forma simples

Uma rede MLP (Perceptron Multicamadas) funciona de maneira semelhante ao cérebro humano, onde "neurônios artificiais" processam informações em camadas.

**Como funciona nosso reconhecimento facial:**

1. **Transformação da imagem em números**:
   - Cada imagem de rosto (128×128 pixels) é convertida em uma longa lista de 49.152 números (128×128×3 canais RGB)
   - É como "achatar" uma foto colorida em uma única fileira de números

2. **Processamento em camadas**:
   - **Primeira camada (2048 neurônios)**: Recebe os 49.152 números e identifica padrões básicos (como bordas, contornos)
   - **Camadas intermediárias (1024→512→128 neurônios)**: Combinam esses padrões em características mais complexas (formato dos olhos, nariz, etc.)
   - **Camada final (número de pessoas)**: Cada neurônio representa uma pessoa. O que "acender mais forte" indica quem é reconhecido

3. **Decisão por "votação"**:
   - Após passar pelos cálculos internos, cada neurônio de saída tem um "valor de ativação"
   - O neurônio com maior valor determina a pessoa reconhecida
   - Esse valor é convertido em percentual de confiança (ex: "João: 92.5%")

**Analogia:** Imagine uma série de filtros cada vez mais específicos. O primeiro filtro separa "é um rosto?" da imagem inicial. Os filtros seguintes procuram características específicas: "tem olhos verdes?", "tem nariz fino?", "tem queixo marcado?". A combinação única dessas características permite identificar a pessoa específica.

**Durante o treinamento**, o sistema aprende automaticamente quais características são mais importantes para distinguir cada pessoa no dataset, ajustando milhões de "pesos" internos que determinam como os padrões são interpretados.

**Na identificação em tempo real**, uma nova imagem percorre o mesmo caminho, e a rede compara seus padrões com o que aprendeu anteriormente.

#### Detalhes da Rede MLP

A parte central do sistema é a Rede Neural Multicamadas (MLP) definida em `src/model/model.py` na classe `FaceRecognitionMLP`. Seus principais aspectos são:

1. **Tamanho da entrada**
   - Cada crop é redimensionado ao tamanho `(H, W)` configurado em `config.yaml` (`image_processing.target_size`).
   - O tensor resultante tem forma `(C, H, W)` e é achatado para `(C*H*W)` antes de entrar no MLP.

2. **Camadas fully-connected**
   - `fc1`: `input_size` → 2048 unidades
   - `fc2`: 2048 → 1024 unidades
   - `fc3`: 1024 → 512 unidades
   - `fc4`: 512 → 128 unidades
   - `fc5`: 128 → `num_classes` (número de pessoas + `unknown`)

3. **Função de ativação**
   - ReLU após cada uma das quatro primeiras camadas (`fc1` a `fc4`).
   - Sem ativação na camada de saída (`fc5`), pois usa logits para `softmax`.

4. **Dropout**
   - Adicionado entre as camadas, se `training.dropout_rate > 0`.
   - Taxa configurável via `config.yaml` (`training.dropout_rate`).

5. **Treinamento**
   - **Perda**: `CrossEntropyLoss` padrão.
   - **Otimizador**: Adam, com taxa de aprendizado em `training.learning_rate` (padrão: 0.001).
   - **Early stopping**: Configuraado em `training.early_stopping.patience` (padrão: 7 épocas).
   - **Balanceamento**: Amostragem limitada por classe (max `training.sample_per_class` imagens por pessoa).

6. **Persistência**
   - Ao final do treinamento, `save_model` grava os pesos em `.pth` e a lista de classes em `.pkl`, criando diretórios automaticamente.

7. **Uso em Inferência**
   - Em `src/face/face_recognition.py`, o método `recognize_face` aplica `softmax` sobre os logits para obter probabilidades.
   - Retorna a classe com maior probabilidade e a confiança em porcentagem.

Este design simples e direto permite fácil treinamento com os dados coletados, mantendo boa performance de reconhecimento.

### Avaliação

#### Resultados

![Matriz de Confusão](confusion_matrix.png)

- **Overall accuracy**: 99.33%
- **Detalhes do relatório de classificação**:

```
              precision    recall  f1-score   support

  anabeatriz       0.97      0.93      0.95       112
    anapaula       0.94      1.00      0.97       130
    bernardo       0.87      0.96      0.91       376
       livia       0.89      0.97      0.93       147
     unknown       1.00      0.99      1.00     14573

    accuracy                           0.99     15338
   macro avg       0.93      0.97      0.95     15338
weighted avg       0.99      0.99      0.99     15338
```

- **Acurácia por classe**:
  - anabeatriz: 92.86%
  - anapaula: 100.00%
  - bernardo: 96.28%
  - livia: 96.60%
  - unknown: 99.48%

#### Configurações Utilizadas

As configurações foram carregadas de `config.yaml`:

```yaml
system:
  python_version: "3.11+"
  device: "cuda"

face_detection:
  model_name: "buffalo_s"
  det_size: [640, 640]
  margin: 20

image_processing:
  target_size: [128, 128]
  normalization:
    mean: [0.5]
    std: [0.5]

training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 0.001
  seed: 42
  validation_split: 0.1
  early_stopping:
    enabled: true
    patience: 7

recognition:
  confidence_threshold: 30.0
  looking_detection:
    enabled: true
    threshold: 0.75

camera:
  width: 1280
  height: 720
  fps: 30
```

### Fluxo de Processamento em Tempo Real

O diagrama abaixo mostra as funções e configurações aplicadas em cada etapa do reconhecimento em tempo real:

```text
┌───────────────────────────────────────────────────────────────────────────┐
│ [1] CAPTURA                                                               │
│ - cv2.VideoCapture(0)                                                     │
│ - cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT, config.get('camera.width/height')) │
│ - ret, frame = cap.read()                                                 │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [2] DETECÇÃO                                                              │
│ - detect_faces(frame, face_analyzer)                                      │
│ - InsightFaceFaceAnalysis.get(frame)                                      │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [3] LOOP POR FACE                                                         │
│ for face in faces:                                                        │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [4] EXTRAÇÃO                                                              │
│ - extract_face(frame, face)                                               │
│ - center_crop_around_nose(...) usando face.bbox e face.kps                │
│ - redimensiona para config.get('image_processing.target_size')            │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [5] PRÉ-PROCESSAMENTO                                                     │
│ - PreprocessModule.remove_background_grabcut(face_img)                    │
│   (MediaPipe SelfieSegmentation)                                          │
│ - TransformFactory.preprocess_face_tensor(face_no_bg, device)             │
│   (ToTensor, Normalize(mean,std) do config)                                │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [6] INFERÊNCIA                                                            │
│ - recognize_face(model, face_img, classes, device)                        │
│ - model(face_tensor) + softmax(outputs)                                   │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [7] PÓS-PROCESSAMENTO                                                     │
│ - if confidence < config.get('recognition.confidence_threshold'):        │
│     result['label']='Desconhecido'                                        │
│   else: result['label']=nome + f": {confidence:.1f}%"                   │
│ - escolhe result['color'] conforme config                                │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [8] DESENHO                                                               │
│ - draw_recognition_results(frame, face, result)                           │
│ - draw_stats(frame, fps, process_time, len(faces), extra_info)            │
└───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [9] EXIBIÇÃO                                                              │
│ - cv2.imshow("Face Recognition", frame)                                 │
│ - key = cv2.waitKey(1) para controle de teclas                            │
└───────────────────────────────────────────────────────────────────────────┘
```

**Detalhes de cada etapa:**

1. **Captura**: Frame RGB capturado via OpenCV da webcam (configurável para 1280×720)
2. **Detecção**: InsightFace detecta faces, retornando bounding boxes e landmarks (106 + 5 pontos-chave)
3. **Loop de faces**: Cada face detectada é processada separadamente
4. **Extração**: Recorte da face centralizado no nariz usando distância interpupilar para escala
5. **Pré-processamento**:
   - Remoção de fundo via MediaPipe SelfieSegmentation
   - Conversão para tensor PyTorch
   - Normalização usando médias/desvios do ImageNet
   - Adição da dimensão de lote (batch)
6. **Inferência**: Forward pass no MLP seguido de softmax para obter probabilidades
7. **Pós-processamento**:
   - Verificação contra threshold de confiança (`recognition.confidence_threshold`)
   - Decisão entre "unknown" e pessoa reconhecida
   - Formatação da etiqueta com nome e porcentagem
8. **Visualização**: Desenho de caixa delimitadora, rótulo e informações de estatísticas no frame
9. **Exibição**: Apresentação do frame processado com anotações ao usuário

Este fluxo executa em tempo real em um loop contínuo, processando cada frame capturado.
