# HW 6 Spark Random Hyperplanes LSH

Реализовать Random Hyperplanes LSH (для косинусного расстояния) в виде Spark Estimator:

1. Посмотреть как устроены MinHashLSH и BucketedRandomProjectionLSH в Spark
2. Унаследоваться от LSHModel и LSH
3. Определить недостающие методы для модели (hashFunction, hashDistance keyDistance, write) и для LSH (createRawLSHModel, copy, transformSchema)

Дополнительное задание (30 баллов):

1. Сделать предсказания (на тех же данных и фичах: HashingTf-Idf)
2. Подобрать количество гиперплоскостей и трешхолд по расстоянию

## Результат запуска тестов

Также тесты запускаются в GitHub Actions. См. badge ниже со статусом:

[![CI](https://github.com/KernelA/made-bd-hw6/actions/workflows/test.yaml/badge.svg)](https://github.com/KernelA/made-bd-hw6/actions/workflows/test.yaml)

## Реализация

### Основное задание

[Реализация LSH](src/main/scala/org/apache/spark/ml/made/CosineRandomHyperplanesLSH.scala)

### Дополнительное задание

[Подбор числа гиперплоскостей и порога](report/main.log)

## Как запустить:

[Скачать данные c Kaggle](https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews)

Разархивировать и положить в `data`.

Установить SBЕ и Java.

На Windows необходимо установить [Hadoop и HDFS](https://towardsdatascience.com/installing-hadoop-3-2-1-single-node-cluster-on-windows-10-ac258dd48aef)

Выполнить команду:
```
sbt run
```

Запуск тестов:
```
sbt test
```
