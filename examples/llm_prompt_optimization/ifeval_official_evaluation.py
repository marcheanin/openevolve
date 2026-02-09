"""
ОБЯЗАТЕЛЬНАЯ интеграция с официальным скриптом оценки IFEval от Google.

Этот модуль ТОЛЬКО использует официальный скрипт от Google:
https://github.com/google-research/google-research/tree/master/instruction_following_eval

Если официальный скрипт недоступен - выбрасывает ошибку и завершает выполнение.
"""

import os
import sys
from typing import Dict, List, Tuple
import dataclasses

# Глобальные переменные для официальных модулей
evaluation_lib = None
instructions_registry = None
OFFICIAL_EVAL_INITIALIZED = False

# Список необходимых ресурсов NLTK для IFEval
NLTK_RESOURCES = ['punkt_tab', 'punkt', 'averaged_perceptron_tagger']


def ensure_nltk_resources():
    """Убеждается, что все необходимые ресурсы NLTK загружены."""
    try:
        import nltk
        
        resources_to_download = {
            'punkt_tab': ['tokenizers/punkt_tab', 'punkt_tab'],
            'punkt': ['tokenizers/punkt/english.pickle', 'punkt'],
            'averaged_perceptron_tagger': ['taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger']
        }
        
        for resource_name, (resource_path, download_name) in resources_to_download.items():
            try:
                # Пробуем найти ресурс
                nltk.data.find(resource_path)
            except (LookupError, OSError):
                print(f"Загружаю ресурс NLTK: {download_name}...")
                try:
                    nltk.download(download_name, quiet=True)
                    print(f"✓ Ресурс {download_name} загружен")
                except Exception as e:
                    print(f"⚠ Не удалось загрузить {download_name}: {e}")
    except Exception as e:
        print(f"Предупреждение: Не удалось проверить/загрузить ресурсы NLTK: {e}")


def initialize_official_evaluation(eval_path: str = None) -> None:
    """
    Инициализирует официальный скрипт оценки Google.
    ВЫБРАСЫВАЕТ ИСКЛЮЧЕНИЕ, если скрипт недоступен.
    
    Args:
        eval_path: Путь к директории instruction_following_eval
        
    Raises:
        RuntimeError: Если официальный скрипт не найден или не может быть импортирован
    """
    global evaluation_lib, instructions_registry, OFFICIAL_EVAL_INITIALIZED
    
    if OFFICIAL_EVAL_INITIALIZED:
        return
    
    # Формируем список путей для поиска
    paths_to_try = []
    
    if eval_path:
        paths_to_try.append(eval_path)
    
    # Определяем базовую директорию (где находится этот файл)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Стандартные пути относительно разных точек запуска
    cwd = os.getcwd()
    paths_to_try.extend([
        os.environ.get("IFEVAL_EVAL_PATH"),
        # Относительно текущего файла (ifeval_official_evaluation.py находится в llm_prompt_optimization/)
        os.path.join(base_dir, "ifeval_experiment", "google-research", "instruction_following_eval"),
        # Относительно рабочей директории (может быть ifeval_experiment/)
        os.path.join(cwd, "google-research", "instruction_following_eval"),
        os.path.join(cwd, "ifeval_experiment", "google-research", "instruction_following_eval"),
        # Если запускаемся из ifeval_experiment/
        os.path.join(cwd, "..", "ifeval_experiment", "google-research", "instruction_following_eval"),
        # Стандартные системные пути
        os.path.expanduser("~/google-research/instruction_following_eval"),
        os.path.expanduser("~/google_research/instruction_following_eval"),
        os.path.abspath("./google-research/instruction_following_eval"),
        os.path.abspath("../google-research/instruction_following_eval"),
        # Относительно корня проекта
        os.path.join(base_dir, "google-research", "instruction_following_eval"),
    ])
    
    # Убираем None и несуществующие пути
    paths_to_try = [p for p in paths_to_try if p and os.path.exists(p)]
    
    # Пробуем импортировать из каждого пути
    for path in paths_to_try:
        if not os.path.exists(path):
            continue
        
        try:
            # Структура репозитория Google: модули используют импорты вида
            # "from instruction_following_eval import instructions_registry"
            # Поэтому нужно добавить РОДИТЕЛЬСКУЮ директорию в sys.path,
            # а сама instruction_following_eval будет пакетом
            
            parent_dir = os.path.dirname(path)  # Родительская директория (google-research/)
            eval_dir = path  # Сама директория instruction_following_eval
            
            # Проверяем, что это действительно та директория
            if not os.path.basename(eval_dir) == "instruction_following_eval":
                continue
            
            eval_lib_file = os.path.join(eval_dir, "evaluation_lib.py")
            if not os.path.exists(eval_lib_file):
                continue
            
            # Добавляем родительскую директорию в sys.path
            # Теперь можно импортировать: from instruction_following_eval import ...
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            try:
                # Модули Google используют импорты вида:
                # from instruction_following_eval import instructions_registry
                # Поэтому нужно создать пакет instruction_following_eval программно
                import types
                import importlib.util
                
                package_name = "instruction_following_eval"
                
                # Создаём пакет в памяти
                if package_name not in sys.modules:
                    package_module = types.ModuleType(package_name)
                    package_module.__path__ = [eval_dir]  # Указываем путь к пакету
                    sys.modules[package_name] = package_module
                
                # Импортируем instructions.py сначала (зависимость для instructions_registry)
                inst_file = os.path.join(eval_dir, "instructions.py")
                if os.path.exists(inst_file):
                    spec_inst = importlib.util.spec_from_file_location(
                        f"{package_name}.instructions", inst_file
                    )
                    if spec_inst is None:
                        raise ImportError(f"Could not create spec for {inst_file}")
                    module_inst = importlib.util.module_from_spec(spec_inst)
                    sys.modules[f"{package_name}.instructions"] = module_inst
                    spec_inst.loader.exec_module(module_inst)
                
                # Импортируем instructions_registry.py
                reg_file = os.path.join(eval_dir, "instructions_registry.py")
                spec_reg = importlib.util.spec_from_file_location(
                    f"{package_name}.instructions_registry", reg_file
                )
                if spec_reg is None:
                    raise ImportError(f"Could not create spec for {reg_file}")
                module_reg = importlib.util.module_from_spec(spec_reg)
                sys.modules[f"{package_name}.instructions_registry"] = module_reg
                spec_reg.loader.exec_module(module_reg)
                
                # Проверяем, что INSTRUCTION_DICT доступен
                if not hasattr(module_reg, 'INSTRUCTION_DICT'):
                    continue
                
                # Сохраняем глобальные переменные
                globals()['instructions_registry'] = module_reg
                globals()['evaluation_lib'] = None  # Не используем evaluation_lib напрямую
                
                OFFICIAL_EVAL_INITIALIZED = True
                print(f"✓ Official IFEval evaluation loaded from: {eval_dir}")
                
                # Убеждаемся, что все ресурсы NLTK загружены
                ensure_nltk_resources()
                
                return
                
            except Exception as e:
                # Если не получилось, пробуем ещё раз с отладочной информацией
                import traceback
                error_details = traceback.format_exc()
                # Не выводим ошибку здесь - попробуем другие пути
                continue
                
        except Exception as e:
            continue
    
    # Если ничего не сработало - ОШИБКА
    checked_paths_str = "\n".join(f"  - {p}" for p in paths_to_try) if paths_to_try else "  (пути не найдены)"
    
    error_msg = f"""
================================================================================
ОШИБКА: Официальный скрипт оценки IFEval от Google не найден!

Чтобы использовать IFEval, необходимо установить официальный скрипт оценки:

1. Клонируйте репозиторий Google:
   git clone https://github.com/google-research/google-research.git

2. Перейдите в директорию:
   cd google-research/instruction_following_eval

3. Установите зависимости:
   pip install -r requirements.txt

4. Укажите путь через переменную окружения:
   export IFEVAL_EVAL_PATH="/path/to/google-research/instruction_following_eval"

   Или поместите репозиторий в:
   openevolve/examples/llm_prompt_optimization/ifeval_experiment/google-research/instruction_following_eval

Проверенные пути:
{checked_paths_str}

Текущая рабочая директория: {os.getcwd()}
Директория этого файла: {base_dir}
================================================================================
"""
    
    raise RuntimeError(error_msg)


def evaluate_ifeval_official_only(
    output: str,
    instruction_text: str,
    instruction_id_list: List[str],
    kwargs: List[Dict] = None,
    eval_path: str = None
) -> Tuple[bool, Dict]:
    """
    Оценка IFEval ТОЛЬКО через официальный скрипт Google.
    
    Использует логику из test_instruction_following_strict:
    - Для каждого instruction_id создаёт соответствующий checker
    - Проверяет все ограничения
    - Возвращает True только если ВСЕ ограничения выполнены
    
    Args:
        output: Ответ модели
        instruction_text: Инструкция из датасета (поле "prompt")
        instruction_id_list: Список ID ограничений (например, ['punctuation:no_comma'])
        kwargs: Параметры для каждой инструкции (соответствует порядку в instruction_id_list)
        eval_path: Путь к официальному скрипту (опционально)
        
    Returns:
        Tuple (passed: bool, details: Dict)
        
    Raises:
        RuntimeError: Если официальный скрипт недоступен
    """
    # Инициализация - выбросит ошибку если недоступен
    initialize_official_evaluation(eval_path)
    
    if instructions_registry is None:
        raise RuntimeError("Official IFEval evaluation modules are None after initialization")
    
    if not instruction_id_list:
        raise RuntimeError(
            "instruction_id_list is empty. Cannot evaluate without constraint IDs."
        )
    
    # Если kwargs не передан, создаём пустые словари
    if kwargs is None:
        kwargs = [{}] * len(instruction_id_list)
    
    if len(kwargs) != len(instruction_id_list):
        raise RuntimeError(
            f"kwargs length ({len(kwargs)}) does not match instruction_id_list length ({len(instruction_id_list)}). "
            f"instruction_id_list: {instruction_id_list}, kwargs: {kwargs}"
        )
    
    # Проверяем, что все kwargs - это словари
    for i, kw in enumerate(kwargs):
        if not isinstance(kw, dict):
            raise RuntimeError(
                f"kwargs[{i}] должен быть словарём, но получен {type(kw)}: {kw}. "
                f"Соответствующий instruction_id: {instruction_id_list[i] if i < len(instruction_id_list) else 'N/A'}"
            )
    
    # Проверяем каждое ограничение по логике из test_instruction_following_strict
    is_following_list = []
    constraint_results = {}
    
    for index, instruction_id in enumerate(instruction_id_list):
        try:
            # Получаем класс checker из регистра
            if instruction_id not in instructions_registry.INSTRUCTION_DICT:
                raise RuntimeError(
                    f"Unknown instruction_id: {instruction_id}. "
                    f"Available: {list(instructions_registry.INSTRUCTION_DICT.keys())[:5]}..."
                )
            
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)
            
            # Получаем kwargs для этой инструкции
            instruction_kwargs = kwargs[index] if index < len(kwargs) else {}
            
            # Проверяем, что kwargs - это словарь
            if not isinstance(instruction_kwargs, dict):
                raise RuntimeError(
                    f"kwargs[{index}] для instruction_id '{instruction_id}' должен быть словарём, "
                    f"но получен: {type(instruction_kwargs)}"
                )
            
            # Очищаем kwargs от None значений
            # HuggingFace dataset может заполнять все поля None, нужно оставить только реальные значения
            cleaned_kwargs = {k: v for k, v in instruction_kwargs.items() if v is not None}
            
            # Инициализируем инструкцию с параметрами
            # Используем ту же логику, что и в оригинальном коде Google (evaluation_lib.py)
            # Передаём только очищенные kwargs (без None значений)
            try:
                if cleaned_kwargs:
                    instruction.build_description(**cleaned_kwargs)
                else:
                    # Если все значения None, вызываем без аргументов (для инструкций типа CommaChecker)
                    instruction.build_description()
            except TypeError as e:
                # Если ошибка типа, возможно kwargs не соответствуют инструкции
                error_str = str(e)
                # Выводим отладочную информацию для диагностики
                print(f"DEBUG: Ошибка при инициализации инструкции '{instruction_id}' (index {index})")
                print(f"DEBUG: исходные kwargs[{index}] = {instruction_kwargs}")
                print(f"DEBUG: очищенные kwargs = {cleaned_kwargs}")
                print(f"DEBUG: instruction_id_list[{index}] = {instruction_id_list[index] if index < len(instruction_id_list) else 'OUT OF RANGE'}")
                print(f"DEBUG: Все instruction_id_list = {instruction_id_list}")
                raise RuntimeError(
                    f"Ошибка при инициализации инструкции '{instruction_id}' (index {index}) с kwargs {cleaned_kwargs}: {error_str}. "
                    f"Проверьте, что kwargs соответствуют ожидаемым параметрам для этой инструкции. "
                    f"Длины: kwargs={len(kwargs)}, instruction_id_list={len(instruction_id_list)}"
                ) from e
            
            # Если инструкция требует промпт в аргументах (как в оригинальном коде)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=instruction_text)
            
            # Проверяем, выполнена ли инструкция
            if output.strip() and instruction.check_following(output):
                is_following_list.append(True)
                constraint_results[instruction_id] = "PASS"
            else:
                is_following_list.append(False)
                constraint_results[instruction_id] = "FAIL"
                
        except RuntimeError:
            # Пробрасываем RuntimeError дальше без изменений
            raise
        except LookupError as e:
            # Ошибка отсутствия ресурса NLTK - пытаемся автоматически загрузить
            error_msg = str(e)
            error_lower = error_msg.lower()
            
            if "punkt_tab" in error_lower or "punkt" in error_lower or "nltk" in error_lower:
                # Пытаемся загрузить недостающий ресурс
                try:
                    import nltk
                    resources_downloaded = []
                    
                    if "punkt_tab" in error_lower:
                        print("Загружаю недостающий ресурс NLTK: punkt_tab...")
                        nltk.download('punkt_tab', quiet=True)
                        resources_downloaded.append('punkt_tab')
                    
                    if "punkt" in error_lower:
                        print("Загружаю недостающий ресурс NLTK: punkt...")
                        nltk.download('punkt', quiet=True)
                        resources_downloaded.append('punkt')
                    
                    if resources_downloaded:
                        print(f"✓ Ресурсы NLTK загружены: {', '.join(resources_downloaded)}")
                    
                    # Пробуем ещё раз после загрузки
                    if output.strip() and instruction.check_following(output):
                        is_following_list.append(True)
                        constraint_results[instruction_id] = "PASS"
                    else:
                        is_following_list.append(False)
                        constraint_results[instruction_id] = "FAIL"
                    continue  # Переходим к следующей инструкции
                    
                except Exception as e2:
                    raise RuntimeError(
                        f"Ошибка NLTK при оценке инструкции '{instruction_id}': {error_msg}. "
                        f"Не удалось автоматически загрузить ресурсы NLTK. "
                        f"Попробуйте установить их вручную: "
                        f"python -c \"import nltk; nltk.download('punkt_tab'); nltk.download('punkt')\""
                    ) from e
            else:
                # Другие LookupError - пробрасываем дальше
                raise
        except Exception as e:
            error_msg = f"Error evaluating constraint {instruction_id} at index {index}: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(f"DEBUG: instruction_id_list length: {len(instruction_id_list)}, kwargs length: {len(kwargs)}")
            if index < len(kwargs):
                print(f"DEBUG: kwargs[{index}] = {kwargs[index]}")
            raise RuntimeError(
                f"Official IFEval evaluation failed: {error_msg}. "
                f"Please check that the official script is correctly installed and kwargs match instruction_id_list."
            ) from e
    
    # Успех только если ВСЕ ограничения выполнены
    all_passed = all(is_following_list)
    
    details = {
        "passed_constraints": sum(is_following_list),
        "total_constraints": len(instruction_id_list),
        "constraint_results": constraint_results,
        "using_official": True,
    }
    
    return all_passed, details
