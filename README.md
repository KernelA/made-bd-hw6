# HW 5 Spark Линейная регрессия на Scala



[Описание задания (129 слайд)](https://github.com/netcitizenrus/MADE_BigData_2021/blob/540e164ecc574b52ca6ffa40636b8728af28aa80/SparkML/Distiributed.pdf)


## Результат работы

[Логи обучения](report/main.log)

## Резлультаты запуска тестов

[Лог](report/test.log)

Также тесты запускаются в GitHub Actions. См. badge ниже со статусом:

[![CI](https://github.com/KernelA/made-bd-hw5/actions/workflows/test.yaml/badge.svg)](https://github.com/KernelA/made-bd-hw5/actions/workflows/test.yaml)

## Реализация

[Реализация линейной регрессии](src/main/scala/org/apache/spark/ml/made/LinearRegression.scala)

## Как запустить:

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
