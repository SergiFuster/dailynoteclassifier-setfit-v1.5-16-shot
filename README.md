# SetFit with BAAI/bge-small-en-v1.5

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 6 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label        | Examples                                                                                                                                                                                                                              |
|:-------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| cita         | <ul><li>'Cita para la vacuna de refuerzo el martes por la mañana.'</li><li>'Reservar un restaurante para celebrar el aniversario de mis padres.'</li><li>'Cita de revisión dental para limpieza el jueves a las 11:00 a.m.'</li></ul> |
| recordatorio | <ul><li>'Recordar ver el documental sobre cambio climático el viernes.'</li><li>'Recordar regar las plantas en el jardín dos veces por semana.'</li><li>'Llamar a mi hermana para felicitarla por su nuevo trabajo.'</li></ul>        |
| comprar      | <ul><li>'Comprar una botella de agua para llevar al trabajo.'</li><li>'Comprar vitaminas y suplementos para el mes.'</li><li>'Comprar repelente de mosquitos para el viaje al campo.'</li></ul>                                       |
| estudios     | <ul><li>'Leer el capítulo 5 del libro de historia antes del jueves.'</li><li>'Estudiar el tema de estadística para el examen de la próxima semana.'</li><li>'Preparar el resumen del capítulo para la discusión en clase.'</li></ul>  |
| trabajo      | <ul><li>'Revisar y firmar los documentos para la solicitud de préstamo.'</li><li>'Programar una reunión con el equipo para revisar avances.'</li><li>'Actualizar el reporte semanal con los últimos datos de rendimiento.'</li></ul>  |
| hogar        | <ul><li>'Limpiar y organizar el jardín el domingo en la mañana.'</li><li>'Organizar el escritorio y archivar documentos importantes.'</li><li>'Lavar y guardar la ropa de esta semana.'</li></ul>                                     |

## Evaluation

### Metrics
| Label   | Accuracy |
|:--------|:---------|
| **all** | 0.6875   |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("Comprar una botella de agua para llevar al trabajo.")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 6   | 9.7912 | 15  |

| Label        | Training Sample Count |
|:-------------|:----------------------|
| cita         | 15                    |
| comprar      | 16                    |
| trabajo      | 16                    |
| recordatorio | 12                    |
| estudios     | 16                    |
| hogar        | 16                    |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (50, 50)
- max_steps: 20
- sampling_strategy: oversampling
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch | Step | Training Loss | Validation Loss |
|:-----:|:----:|:-------------:|:---------------:|
| 0.05  | 1    | 0.1978        | -               |

### Framework Versions
- Python: 3.11.5
- SetFit: 1.1.0
- Sentence Transformers: 3.3.1
- Transformers: 4.42.2
- PyTorch: 2.5.1+cpu
- Datasets: 3.1.0
- Tokenizers: 0.19.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
