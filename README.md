# HW 3 Линейная регрессия на Scala

[Описание задания](https://github.com/netcitizenrus/MADE_BigData_2021/blob/3a0272a6cb3d8ea626e360668fbeaa1c794b5163/HW3_Scala.pdf)

Реализована простая линейная регрессия на основе МНК без регуляризации.

## Данные

Данные были синтетически сгенерированы для простой проверки насколько алгоритм правильно работает. К правильным результатам был добавлен шум.

* [train](data/train.csv)
* [test](data/test.csv)
* [Предсказание на test](data/predicted.csv)
* [Настоящие значения параметров линейно модели](data/true_coeff.csv)

## Результат обучения

[Логи обучения с расчётом ошибок на train в ходе кросс-валидации](train_log/train.log)

## Реализация

[Реализация линейной регрессии](src/main/scala/Main.scala)

## Как запустить:

Установить SBЕ и Java.

Выполнить команду:
```
sbt "run --train ./data/train.csv --test ./data/test.csv --out ./data/predicted.csv --true ./data/true_coeff.csv"
```
