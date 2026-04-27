# Локальный чек-лист проверки проекта

1. Скачать репозиторий или выполнить `git pull`.
2. Создать виртуальное окружение:
   - `python -m venv .venv`
3. Активировать окружение:
   - Windows: `.venv\Scripts\activate`
4. Установить зависимости:
   - `pip install -r requirements.txt`
5. Выполнить компиляционную проверку:
   - `python -m py_compile activation_params.py activation_core.py activation_plots.py activation_report.py activation_validation.py activation_parametric.py app.py test_activation_smoke.py test_imports.py`
6. Запустить smoke-test:
   - `python test_activation_smoke.py`
7. Сравнить результаты с контрольными значениями v3.2.
8. Запустить приложение:
   - `streamlit run app.py`
9. Проверить в UI:
   - выбор 7 видов сырья;
   - одиночный расчёт;
   - поля реактора;
   - параметрический анализ;
   - оптимизацию;
   - экспорт DOCX/TXT/CSV/PNG.
10. Если smoke-test не проходит:
    - не править физику сразу;
    - проверить единицы измерения;
    - проверить ключи `params`;
    - проверить `default params`;
    - проверить сетку расчёта;
    - сравнить с legacy-файлом.
11. Откат через GitHub/PR:
    - открыть PR;
    - использовать `Revert` на merge-коммите или `git revert <commit>`;
    - повторно прогнать smoke-test и компиляцию.
