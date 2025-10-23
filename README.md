# Scientific Image Forgery Detection

Este projeto contém minha submissão para o desafio "Scientific Image Forgery Detection" (Recod.ai/LUC | Kaggle). O objetivo é detectar falsificações em imagens científicas usando um modelo de visão computacional customizado, construído em PyTorch.

**Competition:** [Recod.ai/LUC - Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection)

## 🎯 Objetivo

Detectar falsificações em imagens científicas utilizando técnicas de deep learning e visão computacional. O projeto visa identificar manipulações em imagens científicas que podem comprometer a integridade da pesquisa.

## 🏗️ Estrutura do Projeto

```
.
├── .devcontainer/          # Configuração do devcontainer para Codespaces
├── data/                   # Dados do Kaggle
│   ├── raw/               # Dados brutos baixados
│   ├── processed/         # Dados processados
│   └── external/          # Dados externos
├── notebooks/             # Jupyter notebooks para exploração
│   └── 01_data_exploration.ipynb
├── src/                   # Código fonte
│   └── forgery_detection/ # Módulo principal
│       ├── __init__.py
│       ├── data.py       # Carregamento e preprocessamento de dados
│       ├── model.py      # Definições de modelos
│       └── train.py      # Utilitários de treinamento
├── tests/                 # Testes unitários
├── models/                # Modelos treinados salvos
├── pyproject.toml         # Configuração do projeto e dependências
└── README.md

```

## 🚀 Getting Started

### Pré-requisitos

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (gerenciador de pacotes Python)
- GPU com CUDA (recomendado para treinamento)

### Configuração com GitHub Codespaces

Este projeto está configurado para funcionar perfeitamente com GitHub Codespaces usando a máquina mais potente (32-core):

1. Abra o repositório no GitHub
2. Clique em "Code" > "Codespaces" > "Create codespace on main"
3. O devcontainer será automaticamente configurado com:
   - Python 3.11
   - uv package manager
   - Todas as extensões necessárias do VS Code
   - 32 cores e 64GB de memória

### Instalação Local

1. Clone o repositório:
```bash
git clone https://github.com/giovanimds/kaggle---Recod.ai-LUC---Scientific-Image-Forgery-Detection.git
cd kaggle---Recod.ai-LUC---Scientific-Image-Forgery-Detection
```

2. Instale o uv (se ainda não tiver):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Instale as dependências do projeto:
```bash
uv pip install -e .
```

4. Para desenvolvimento, instale as dependências opcionais:
```bash
uv pip install -e ".[dev]"
```

### Download dos Dados

1. Instale a Kaggle CLI:
```bash
uv pip install kaggle
```

2. Configure suas credenciais do Kaggle (`~/.kaggle/kaggle.json`)

3. Baixe os dados da competição:
```bash
kaggle competitions download -c recodai-luc-scientific-image-forgery-detection
unzip recodai-luc-scientific-image-forgery-detection.zip -d data/raw/
```

## 📊 Exploração de Dados

Execute o notebook de exploração:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 🧪 Treinamento

```python
from forgery_detection.model import create_model
from forgery_detection.data import ForgeryDetectionDataset, get_transforms
from forgery_detection.train import train_epoch, evaluate

# Criar modelo
model = create_model(model_name="efficientnet_b0", num_classes=2)

# Preparar dados
train_dataset = ForgeryDetectionDataset(
    data_dir="data/raw/train",
    transform=get_transforms("train")
)

# Treinar
# ... (código de treinamento)
```

## 🛠️ Desenvolvimento

### Linting e Formatação

```bash
# Formatar código com black
black src/ tests/

# Lint com ruff
ruff check src/ tests/

# Type checking com mypy
mypy src/
```

### Testes

```bash
pytest tests/ -v --cov=src
```

## 📦 Dependências Principais

- **PyTorch**: Framework de deep learning
- **torchvision**: Visão computacional
- **timm**: Modelos pré-treinados
- **albumentations**: Augmentação de dados
- **OpenCV**: Processamento de imagens
- **Jupyter**: Notebooks interativos

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, sinta-se à vontade para abrir issues ou pull requests.

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🔗 Links Úteis

- [Página da Competição](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection)
- [Descrição dos Dados](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection/data)
- [Documentação do PyTorch](https://pytorch.org/docs/)
- [Documentação do timm](https://timm.fast.ai/)
