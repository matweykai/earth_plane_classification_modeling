# Эксперименты
Здесь описываются эксперименты, которые проводились с моделью и данными

* [ClearML Project](https://app.clear.ml/projects/039f9b58f33b46a2b9a00150f8320bd9/experiments/86ccd1d7b3d940dda92f87d96c44aa92/execution?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=)

---

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
