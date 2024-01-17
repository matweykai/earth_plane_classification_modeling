# Эксперименты
Здесь описываются эксперименты, которые проводились с моделью и данными

* [ClearML Project](https://app.clear.ml/projects/039f9b58f33b46a2b9a00150f8320bd9/experiments/86ccd1d7b3d940dda92f87d96c44aa92/execution?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=)


## 12.09.2023 Получение Baseline модели
Модель была запущена с дефолтными настройками и получены метрики, которые будем в дальнейшем улучшать

```
TEST:
    f1: 0.57
    precision: 0.62
    recall: 0.55
    AP: 0.62

VAL:
    bce_loss: 0.1569
    f1: 0.57
    precision: 0.66
    recall: 0.53
    AP: 0.61
```
Тест и валидация достаточно хорошо сходятся

[ClearML run](https://app.clear.ml/projects/039f9b58f33b46a2b9a00150f8320bd9/experiments/86ccd1d7b3d940dda92f87d96c44aa92/output/metrics/scalar?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=)


## 12.09.2023 Использование exponential lr
Для loss кривой попробуем использовать другой LRScheduler - Exponential LR
Он снижает скорость обучения постепенно, что должно положительно сказаться на обучении модели

```
TEST:
    f1: 0.59 (+0.02)
    precision: 0.64 (+0.02)
    recall: 0.57 (+0.02)
    AP: 0.65 (+0.03)

VAL:
    bce_loss: 0.1291
    f1: 0.58 (-0.01)
    precision: 0.62 (-0.04)
    recall: 0.57 (-0.04)
    AP: 0.62 (+0.01)
```

ClearML:

* [gamma = 0.95](https://app.clear.ml/projects/039f9b58f33b46a2b9a00150f8320bd9/compare-experiments;ids=10b36e5579df4f6799e2da992d64cb6e,86ccd1d7b3d940dda92f87d96c44aa92/scalars/graph?scalars=graph) - лучше чем baseline

* [gamma = 0.90](https://app.clear.ml/projects/039f9b58f33b46a2b9a00150f8320bd9/compare-experiments;ids=10b36e5579df4f6799e2da992d64cb6e,85ffd4ac40e94789b41002cad4b23e10/scalars/graph) - хуже чем предыдущий вариант

* [gamma = 0.85](https://app.clear.ml/projects/039f9b58f33b46a2b9a00150f8320bd9/compare-experiments;ids=85ffd4ac40e94789b41002cad4b23e10,10b36e5579df4f6799e2da992d64cb6e,7d2b949192c44b1f995553b7ce8c9584/scalars/graph) - лучше предыдущих вариантов (по AP) (выбран как самый лучший)

* [gamma = 0.80](https://app.clear.ml/projects/039f9b58f33b46a2b9a00150f8320bd9/compare-experiments;ids=064943fb4d3a4a6ba397a833991e4b71,7d2b949192c44b1f995553b7ce8c9584,10b36e5579df4f6799e2da992d64cb6e/scalars/graph?scalars=graph) - на уровне с бейзлайном


## 13.09.2023 Подбор размера модели
Попробуем использовать более сложные модели для данной задачи

```
TEST:
    f1: 0.59 (+0.02)
    precision: 0.72 (+0.10)
    recall: 0.55 (+0.0)
    AP: 0.64 (+0.02)

VAL:
    bce_loss: 0.2115
    f1: 0.60 (+0.03)
    precision: 0.71 (+0.05)
    recall: 0.56 (+0.03)
    AP: 0.63 (+0.02)
```

ClearML:
- [ResNet152 (gamma = 0.8)](https://app.clear.ml/projects/039f9b58f33b46a2b9a00150f8320bd9/compare-experiments;ids=7d2b949192c44b1f995553b7ce8c9584,dbc13a2b23334e8f8f18ce92b1f6c89a/scalars/graph?scalars=graph) - AP остался примерно таким же, но время обучения увеличилось очень сильно (было 28 мин, а стало ~3 часа)

- [ResNet50 (gamma = 0.8)](https://app.clear.ml/projects/039f9b58f33b46a2b9a00150f8320bd9/compare-experiments;ids=1b192b456db34dbaab912f29321d6d16,7d2b949192c44b1f995553b7ce8c9584/scalars/graph?scalars=graph) - случайно указал не ту gamma, но AP вырос, f1 тоже, а hamming уменьшился

- [ResNet50 (gamma = 0.85)](https://app.clear.ml/projects/039f9b58f33b46a2b9a00150f8320bd9/experiments/a35b3b74adf74fdf89ae02b3a82c9531/output/metrics/scalar) - выбираем resnet50, тк она обладает хорошим качеством и быстро обучается


## 13.09.2023 Focal loss + BCE loss
Используем взвешивание loss'a на основе вероятностных меток

[Результаты](https://app.clear.ml/projects/039f9b58f33b46a2b9a00150f8320bd9/experiments/98cad6755087468d8cdb75ed65c503d6/execution?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=) - **Данный эксперимент дал очень плохие результаты** 

## 14.09.2023 Weighted BCE
Взвешивание классов производилось по их априорному распределению

```
TEST:
    f1: 0.53 (-0.04)
    precision: 0.63 (+0.01)
    recall: 0.47 (-0.08)
    AP: 0.62 (-)

VAL:
    weighted_bce_loss: 0.0015
    f1: 0.52 (-0.05)
    precision: 0.65 (-0.01)
    recall: 0.47 (-0.06)
    AP: 0.59 (-0.02)
```

Как видно по метрикам данный подход нам не очень помог
[Результаты](https://app.clear.ml/projects/039f9b58f33b46a2b9a00150f8320bd9/compare-experiments;ids=4746c43b1dca4d82829cf0b2d08402e5,a35b3b74adf74fdf89ae02b3a82c9531/scalars/graph?scalars=graph)