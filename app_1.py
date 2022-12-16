#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from random import randint

import pickle


PATH_TPL = ('X_bp.xlsx', 'X_nup.xlsx')

TARGET_VARS = ['Модуль упругости при растяжении, ГПа', 'Прочность при растяжении, МПа']
TARGET_FOR_NN = 'Соотношение матрица-наполнитель'


def data_parser(path_list, joint_type='inner'):
    data = pd.DataFrame()
    for i, path in enumerate(path_list):
        df_part = pd.read_excel(path_list[i], index_col=0)
        print(f'Исходные данные ({i + 1} часть) взяты из {path}, '
              f'размерность {i + 1} части данных =', df_part.shape)
        data = data.join(df_part, how=(joint_type if i > 0 else 'outer'))
    print(f'Данные объединены по индексу, тип объединения {joint_type}, '
          f'размерность объединенного датасета = {data.shape}')
    columns_ending_with_targets = list(data.columns)
    for col in TARGET_VARS:
        columns_ending_with_targets.remove(col)
        columns_ending_with_targets.append(col)
    return data.reindex(columns=columns_ending_with_targets)


def choise_frome_set():
    print('Вы ввели значение "-1", это означает, что параметры не будут вводиться,'
          'а будут взяты из имеющегося набора данных.')
    for _ in range(5):
        try:
            string_num = int(input(f'Введите номер (индекс) строки от 0 до 1023: '))
            break
        except:
            print('Требуется ввести целое число. Попробуйте ещё раз.')
    else:
        string_num = randint(0, 1023)
        print(f'Вы сделали 5 некорректных попыток ввода, будет взято случайное число = {string_num}')
    if string_num not in range(0, 1023):
        print(f'В датасете нет строки {string_num}')
        string_num = randint(0, 1023)
        print(f'Будет взято случайное число = {string_num}')
    df = data_parser(PATH_TPL)
    return df.loc[string_num:string_num]


def ml_prediction():
    print('''
    Для получения прогноза модуля упругости при растяжении и 
    прочности при растяжении введите исходные 11 параметров композитного материала:
    ''')
    X_cast = pd.DataFrame([], columns=['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
                                       'модуль упругости, ГПа', 'Количество отвердителя, м.%',
                                       'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
                                       'Поверхностная плотность, г/м2', 'Потребление смолы, г/м2',
                                       'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки',
                                       'Модуль упругости при растяжении, ГПа', 'Прочность при растяжении, МПа'])
    X_lst = []
    for param in X_cast.columns[:11]:
        for _ in range(10):
            try:
                X_lst.append(float(input(f'Введите {param}: ')))
                break
            except:
                print('Требуется ввести вещественное число, соответствующее параметру. Попробуйте ещё раз.')
        else:
            print('Вы сделали 10 некорректных попыток ввода, работа с прогнозом приостановлена.\n\n'
                  '                         возврат в главное меню\n')
            return
        if X_lst[-1] == -1:
            X_cast = choise_frome_set()
            break
    if X_lst[-1] != -1:
        X_cast.loc[1] = X_lst + [0, 0]
    y_cast_test = X_cast[TARGET_VARS]

    print(f'\nПроводится прогноз целевых параметров {TARGET_VARS} для материала со следующими характеристиками:')
    print(X_cast.drop(TARGET_VARS, axis=1).T)
    mod = []
    X_cast[TARGET_VARS[0]] = 0
    X_cast[TARGET_VARS[1]] = 0
    with open(f'scaler.pickle', 'rb') as f:
        mms = pickle.load(f)
    mms.transform(X_cast)
    for i in range(2):
        with open(f'best_for_{i}.pickle', 'rb') as f:
            mod.append(pickle.load(f))
        print(f'Прогноз показателя {TARGET_VARS[i]}, проводится моделью:')
        print(mod[i])
        X_y = np.ones((1, 13))
        X_y[0, 11 + i] = mod[i].predict(X_cast.drop(TARGET_VARS, axis=1))
#         print(X_y)          # отладка
        X_y = mms.inverse_transform(X_y)
#         print(X_y)          # отладка
        print(f'Прогнозное значение показателя {TARGET_VARS[i]} = {X_y[0, 11 + i]:.4f}{TARGET_VARS[i][-4:]}\n\n')
    if X_lst[-1] == -1:
        if input('Показать значения из датасета?  1=да=yes / no=any_key') in ('1', 'yes', 'y', 'да', 'д'):
            print(y_cast_test)
    print('Если хотите продолжить работу с приложением, выберите команду:')


def nn_prediction():
    print('\n\tДля получения соотношения матрица-наполнитель введите исходные 12 параметров композитного материала:\n')
    X_cast = pd.DataFrame([], columns=['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
                                       'модуль упругости, ГПа', 'Количество отвердителя, м.%',
                                       'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
                                       'Поверхностная плотность, г/м2', 'Потребление смолы, г/м2',
                                       'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки',
                                       'Модуль упругости при растяжении, ГПа', 'Прочность при растяжении, МПа'])
    X_lst = []
    for param in X_cast.columns[1:]:
        for _ in range(10):
            try:
                X_lst.append(float(input(f'Введите {param}: ')))
                break
            except:
                print('Требуется ввести вещественное число, соответствующее параметру. Попробуйте ещё раз.')
        else:
            print('Вы сделали 10 некорректных попыток ввода, работа с прогнозом приостановлена.\n\n'
                  '                         возврат в главное меню\n')
            return
        if X_lst[-1] == -1:
            X_cast = choise_frome_set()
            break
    if X_lst[-1] != -1:
        X_cast.loc[1] = [0.] + X_lst
    y_cast_test = X_cast[[TARGET_FOR_NN]]

    print(f'\nПроводится прогноз целевого параметра {TARGET_FOR_NN} для материала со следующими характеристиками:')
    print(X_cast.drop(TARGET_FOR_NN, axis=1).T)
    X_cast[TARGET_FOR_NN] = 0
    with open(f'scaler.pickle', 'rb') as f:
        mms = pickle.load(f)
    mms.transform(X_cast)
    with open(f'best_for_3.pickle', 'rb') as f:
        mod = pickle.load(f)
    print(f'Прогноз показателя {TARGET_FOR_NN}, проводится моделью:')
    print(mod)
    X_y = np.ones((1, 13))
    X_y[0, 0] = mod.predict(X_cast.drop(TARGET_FOR_NN, axis=1))
#     print(X_y)          # отладка
    X_y = mms.inverse_transform(X_y)
#     print(X_y)          # отладка
    print(f'Прогнозное значение показателя {TARGET_FOR_NN} = {X_y[0, 0]:.4f}\n\n')
    if X_lst[-1] == -1:
        if input('Показать значения из датасета?  1=да=yes / no=any_key') in ('1', 'yes', 'y', 'да', 'д'):
            print(y_cast_test)
    print('Если хотите продолжить работу с приложением, выберите команду:')


# Приложение
if __name__ == '__main__':
    for _ in range(10):
        print('''
        Получить прогноз модуля упругости при растяжении и прочности при растяжении - введите "1",
        Получить рекомендацию соотношения матрица-наполнитель - введите "2",
        Завершить работу с приложением - введите "3", или "выход", или "exit", или пустую строку.
        ''')
        choise = input('Введите команду (1, 2 или 3): ')
        if choise.lower() in ('3', 'выход', 'exit', 'e', 'в', ''):
            print('Спасибо за сотрудничесто, приходите ещё.\nВместе мы сделаем этот мир лучше!')
            break
        elif choise[0] == '2':
            nn_prediction()
        elif choise[0] == '1':
            ml_prediction()
        else:
            print('''Требуется ввести цифру от 1 до 3
            You need to enter a number from 1 to 3 for choise (3 for exit), try again''')
    else:
        print('\nВы сделали 10 обращений к программе.\n\n'
              'Спасибо за сотрудничесто, приходите ещё.\nВместе мы сделаем этот мир лучше!')
