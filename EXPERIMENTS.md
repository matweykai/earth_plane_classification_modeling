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