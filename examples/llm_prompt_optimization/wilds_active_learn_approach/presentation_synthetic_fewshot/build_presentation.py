# -*- coding: utf-8 -*-
"""Сборка .pptx презентации (русская версия).

Запуск:
    python build_presentation.py

Результат:
    presentation_synthetic_fewshot.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUT = ROOT / "presentation_synthetic_fewshot_demo_report_iter5_nofulltest.pptx"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

COLOR_TITLE = RGBColor(0x10, 0x20, 0x40)
COLOR_ACCENT = RGBColor(0x1F, 0x77, 0xB4)  # used in footer bar
COLOR_SUB = RGBColor(0x55, 0x55, 0x55)
COLOR_BODY = RGBColor(0x20, 0x20, 0x20)
COLOR_HIGHLIGHT_BG = RGBColor(0xE8, 0xF1, 0xFB)
COLOR_GREEN_BG = RGBColor(0xE6, 0xF4, 0xEA)
COLOR_YELLOW_BG = RGBColor(0xFE, 0xF7, 0xE0)
COLOR_RED_BG = RGBColor(0xFC, 0xE8, 0xE6)
COLOR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def _new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def _add_blank(prs: Presentation):
    layout = prs.slide_layouts[6]
    return prs.slides.add_slide(layout)


def _add_text(slide, x, y, w, h, text, *, size=18, bold=False, color=COLOR_BODY,
              align="left", italic=False):
    from pptx.enum.text import PP_ALIGN
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER,
                       "right": PP_ALIGN.RIGHT}.get(align, PP_ALIGN.LEFT)
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
        run.font.name = "Calibri"
    return tb


def _add_bullets(slide, x, y, w, h, items, *, size=16, color=COLOR_BODY, bullet_char="•"):
    from pptx.enum.text import PP_ALIGN
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(4)
        if isinstance(item, tuple):
            head, tail = item
            run_b = p.add_run()
            run_b.text = f"{bullet_char}  {head}"
            run_b.font.bold = True
            run_b.font.size = Pt(size)
            run_b.font.color.rgb = color
            run_b.font.name = "Calibri"
            run_t = p.add_run()
            run_t.text = f"  {tail}"
            run_t.font.size = Pt(size)
            run_t.font.color.rgb = color
            run_t.font.name = "Calibri"
        else:
            run = p.add_run()
            run.text = f"{bullet_char}  {item}"
            run.font.size = Pt(size)
            run.font.color.rgb = color
            run.font.name = "Calibri"
    return tb


def _add_rect(slide, x, y, w, h, fill=COLOR_HIGHLIGHT_BG, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
    shp.shadow.inherit = False
    return shp


def _add_image(slide, path: Path, x, y, w=None, h=None):
    kw = {}
    if w is not None:
        kw["width"] = Inches(w)
    if h is not None:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def _add_footer(slide, idx_text: str) -> None:
    _add_rect(slide, 0, 7.18, 13.333, 0.32, fill=COLOR_ACCENT)
    _add_text(
        slide, 0.3, 7.22, 10.5, 0.3,
        "Продвинутые методы ML  •  Статья: Intent-based Prompt Calibration (Levi, Brosh, Friedmann), "
        "arXiv:2402.03099",
        size=9, color=RGBColor(0xFF, 0xFF, 0xFF),
    )
    _add_text(slide, 12.3, 7.22, 1.0, 0.3, idx_text, size=10,
              color=RGBColor(0xFF, 0xFF, 0xFF), align="right")


def _add_rating_label(slide, x, y, rating: str, fill) -> None:
    _add_rect(slide, x, y, 0.55, 0.28, fill=fill, line=COLOR_SUB)
    _add_text(slide, x + 0.04, y + 0.03, 0.47, 0.2, rating, size=9, bold=True, align="center")


def _add_example_text(slide, x, y, w, rating, text, fill, *, size=9) -> None:
    _add_rating_label(slide, x, y + 0.02, rating, fill)
    _add_text(slide, x + 0.68, y, w - 0.68, 0.95, text, size=size, color=COLOR_BODY)


def _add_metric_box(slide, x, y, title, value, delta, fill=COLOR_HIGHLIGHT_BG) -> None:
    _add_rect(slide, x, y, 3.0, 1.0, fill=fill, line=COLOR_ACCENT)
    _add_text(slide, x + 0.12, y + 0.10, 2.75, 0.25, title, size=11, bold=True, color=COLOR_TITLE)
    _add_text(slide, x + 0.12, y + 0.38, 2.75, 0.32, value, size=17, bold=True, color=COLOR_ACCENT)
    _add_text(slide, x + 0.12, y + 0.72, 2.75, 0.2, delta, size=9, color=COLOR_SUB)


def slide_title(prs: Presentation) -> None:
    s = _add_blank(prs)
    _add_rect(s, 0, 0, 13.333, 7.5, fill=RGBColor(0xFF, 0xFF, 0xFF))
    _add_rect(s, 0, 2.6, 13.333, 0.05, fill=COLOR_ACCENT)
    _add_text(s, 0.7, 0.85, 12, 0.6,
              "Продвинутые методы ML — доклад по статье",
              size=14, color=COLOR_SUB, italic=True)
    _add_text(s, 0.7, 1.45, 12, 1.05,
              "Синтетические boundary few-shot примеры\nв эволюционной оптимизации промптов",
              size=30, bold=True, color=COLOR_TITLE)
    _add_text(s, 0.7, 2.78, 12, 0.75,
              "По статье: Intent-based Prompt Calibration — Levi, Brosh, Friedmann\n"
              "arXiv:2402.03099 [cs.CL] (5 Feb 2024); код: github.com/Eladlev/AutoPrompt",
              size=15, color=COLOR_SUB)
    _add_text(s, 0.7, 4.05, 12, 0.4,
              "Студент: Андрей Марченко (МФТИ)",
              size=17, color=COLOR_BODY)
    _add_text(s, 0.7, 4.55, 12, 0.45,
              "Тема диплома: доменная оптимизация промптов с активным обучением",
              size=17, color=COLOR_BODY)
    _add_text(s, 0.7, 6.55, 12, 0.4,
              "Весна 2026",
              size=14, color=COLOR_SUB)
    _add_footer(s, "1 / 10")


def slide_intro(prs: Presentation) -> None:
    s = _add_blank(prs)
    _add_text(s, 0.5, 0.28, 12.5, 0.55, "Введение", size=28, bold=True, color=COLOR_TITLE)
    _add_rect(s, 0.5, 0.88, 12.5, 0.04, fill=COLOR_ACCENT)

    _add_text(s, 0.5, 1.0, 12.5, 0.45, "Проблема", size=17, bold=True, color=COLOR_ACCENT)
    _add_bullets(s, 0.5, 1.45, 12.5, 1.55, [
        "Автоматический prompt-engineering часто трактует LLM как «чёрный ящик» и не объясняет причины ошибок.",
        "В реальных данных редко есть идеальные контрастные примеры на границе классов "
        "(например, 4★ vs 5★ с одним явным отличием).",
        "Агрегированные метрики маскируют провалы на трудных подгруппах пользователей.",
    ], size=13)

    _add_text(s, 0.5, 3.15, 12.5, 0.45, "Актуальность", size=17, bold=True, color=COLOR_ACCENT)
    _add_bullets(s, 0.5, 3.58, 12.5, 1.45, [
        "Задача: классификация отзывов Amazon 1–5★ (WILDS), сдвиг по пользователям/поддоменам.",
        "Целевая метрика: R_worst — точность на 10% худших пользователей.",
        "Промпт крайне чувствителен: мелкие правки сдвигают R_worst на порядка 5–15 п.п.",
    ], size=13)

    _add_text(s, 0.5, 5.15, 12.5, 0.45, "Цель собственной работы", size=17, bold=True, color=COLOR_ACCENT)
    _add_bullets(s, 0.5, 5.58, 12.5, 1.25, [
        "Перенести идею boundary cases из IPC в few-shot демонстрации внутри промпта — "
        "генерация и отбор полностью автоматически в цикле активного обучения.",
        "Оценить влияние на R_worst при сопоставимых no-synth абляциях.",
    ], size=13)

    _add_footer(s, "2 / 10")


def slide_article_overview(prs: Presentation) -> None:
    s = _add_blank(prs)
    _add_text(s, 0.5, 0.22, 12.5, 0.55,
              "Обзор статьи — IPC (Intent-based Prompt Calibration)",
              size=22, bold=True, color=COLOR_TITLE)
    _add_rect(s, 0.5, 0.82, 12.5, 0.04, fill=COLOR_ACCENT)

    col_titles = ["Вклад метода", "Механизм (цикл)", "Итоги экспериментов авторов"]
    col_x = [0.45, 4.45, 8.45]
    col_w = 3.95
    for i, t in enumerate(col_titles):
        _add_rect(s, col_x[i], 0.98, col_w, 0.4, fill=COLOR_HIGHLIGHT_BG)
        _add_text(s, col_x[i] + 0.08, 1.02, col_w, 0.35, t, size=14, bold=True, color=COLOR_TITLE)

    _add_bullets(s, col_x[0] + 0.03, 1.45, col_w, 4.35, [
        "IPC калибрует промпт под намерение пользователя, строя маленький «бенчмарк» из синтетических граничных случаев.",
        "Совместно улучшаются промпт и набор сложных примеров; конвейер модульный.",
        "Ориентация на реальные сценарии (модерация, дисбаланс классов) и расширение на генерацию через ranker.",
    ], size=10)

    _add_bullets(s, col_x[1] + 0.03, 1.45, col_w, 4.35, [
        "1) Генератор boundary-примеров (meta-prompt).",
        "2) Оценка промпта на сгенерированных данных; анализ ошибок (confusion matrix) — Analyzer.",
        "3) Генератор нового промпта по истории оценок и анализу.",
        "4) Повтор до стагнации или лимита итераций/стоимости.",
        "Разметка: человек (Argilla) или LLM-estimator.",
    ], size=10)

    _add_bullets(s, col_x[2] + 0.03, 1.45, col_w, 4.35, [
        ("Модели и данные:", "GPT-3.5/4-Turbo; отзывы IMDB как основа."),
        ("Классификация:", "3 бинарные задачи (спойлеры, тональность, PG); сравнение с OPRO и PE; "
         "50 шагов; у IPC — 10 стартовых примеров + синтетика, у бейзлайнов — выборки IMDB. "
         "IPC выше по точности и с меньшей дисперсией (рис. 3, 7); на синт. тесте тональности (300 примеров) разрыв сильнее (рис. 6)."),
        ("Генерация:", "два неоднозначных сценария отзывов; сначала калибруется ranker (50 разметок), затем промпт генерации; "
         "30 итераций ranker+генератор. Средний балл ranker (табл. 2 в arXiv, GPT-4 Turbo): "
         "«энтузиазм и достоверность» — IPC 4.80±0.10 vs initial 4.40±0.05; "
         "«сарказм, но позитив» — IPC 4.92±0.07 vs 4.28±0.02; у части бейзлайнов хуже даже initial prompt из-за дисбаланса реальных рангов (рис. 4)."),
        ("Абляция (§3.3, Tab. 2):", "синтетические данные усиливают качество; компонент Analyzer даёт существенный прирост "
         "(вопреки выводу OPRO о бесполезности явного списка ошибок в meta-prompt)."),
    ], size=9)

    _add_rect(s, 0.5, 5.95, 12.5, 0.95, fill=COLOR_HIGHLIGHT_BG)
    _add_text(s, 0.6, 6.02, 12.3, 0.85,
              "Вывод для моей работы: синтетика на границах — сильный сигнал для калибровки; "
              "в IPC она живёт в контуре оценки и часто опирается на человека или отдельный estimator. "
              "Я переношу идею в контур самого промпта (few-shot) и убираю human-in-the-loop, отдавая отбор MAP-Elites + val.",
              size=11, italic=True, color=COLOR_TITLE)

    _add_footer(s, "3 / 10")


def slide_method(prs: Presentation) -> None:
    s = _add_blank(prs)
    _add_text(s, 0.5, 0.25, 12.5, 0.55,
              "Предлагаемая адаптация — boundary few-shot внутри промпта",
              size=22, bold=True, color=COLOR_TITLE)
    _add_rect(s, 0.5, 0.82, 12.5, 0.04, fill=COLOR_ACCENT)

    _add_image(s, ASSETS / "pipeline_diagram.png", x=0.45, y=0.95, w=8.55)

    _add_rect(s, 9.2, 0.95, 3.85, 3.95, fill=COLOR_HIGHLIGHT_BG)
    _add_text(s, 9.35, 1.0, 3.55, 0.38, "Отличия от IPC", size=14, bold=True, color=COLOR_TITLE)
    _add_bullets(s, 9.28, 1.38, 3.65, 3.45, [
        "Синтетика переходит из контура оценки в контур промпта (in-context).",
        "Нет human-in-the-loop: мутатор MAP-Elites решает, что оставить.",
        "Режимы: replace — жёсткая замена блока; inject_as_hint — мягкая подсказка в system message.",
        "Привязка к confusion matrix и меткам Hard / Anchor в AL-цикле.",
    ], size=10)

    _add_rect(s, 0.5, 5.05, 12.5, 1.95, fill=COLOR_HIGHLIGHT_BG)
    _add_text(s, 0.62, 5.08, 12.2, 0.38,
              "Пример twin-пары (цикл 3, evolve_subsample), граница 4★ / 5★:",
              size=12, bold=True, color=COLOR_TITLE)
    _add_text(s, 0.62, 5.48, 6.15, 1.45,
              "5★ — «This coffee maker is a game-changer! … thermal carafe … easy to clean…»",
              size=10, color=COLOR_BODY)
    _add_text(s, 6.85, 5.48, 6.15, 1.45,
              "4★ — «…really good cup… carafe works well. My main complaint is that the opening to pour water is very small…»",
              size=10, color=COLOR_BODY)

    _add_footer(s, "4 / 10")


def slide_results(prs: Presentation) -> None:
    s = _add_blank(prs)
    _add_text(s, 0.5, 0.22, 12.5, 0.55,
              "Результаты — прирост R_worst и контрольная абляция",
              size=22, bold=True, color=COLOR_TITLE)
    _add_rect(s, 0.5, 0.8, 12.5, 0.04, fill=COLOR_ACCENT)

    _add_image(s, ASSETS / "bar_rworst_pairs.png", x=1.55, y=0.9, w=10.15)

    _add_rect(s, 0.5, 4.95, 6.15, 2.15, fill=COLOR_HIGHLIGHT_BG)
    _add_text(s, 0.62, 5.0, 6.0, 0.38, "Ключевые цифры (test)", size=14, bold=True, color=COLOR_TITLE)
    _add_bullets(s, 0.58, 5.38, 6.05, 1.65, [
        ("Одна категория:", "R_worst 0.503 → 0.569 (+6.6 п.п.)"),
        ("Все 15 категорий:", "R_worst 0.533 → 0.667 (+13.4 п.п.)"),
        ("Combined score:", "0.844 → 0.894 (+5 п.п.)"),
        ("Acc_Hard:", "0.283 → 0.353 (+7 п.п.)"),
    ], size=11)

    _add_rect(s, 6.85, 4.95, 6.15, 2.15, fill=COLOR_HIGHLIGHT_BG)
    _add_text(s, 6.98, 5.0, 6.0, 0.38, "Контроль 8×20 (одна категория)", size=14, bold=True, color=COLOR_TITLE)
    _add_bullets(s, 6.92, 5.38, 6.05, 1.65, [
        "На Acc_Hard синтетика стабильно выше на протяжении всех циклов (+5…+11 п.п.).",
        "R_worst на малом тесте (15 пользователей) зашумлён — для этой пары не делаем вывод о знаке.",
        "Именно это мотивировало масштабирование на все категории.",
    ], size=11)

    _add_footer(s, "5 / 10")


def slide_demo_input_report(prs: Presentation) -> None:
    s = _add_blank(prs)
    _add_text(s, 0.5, 0.25, 12.5, 0.55,
              "Демо-отчёт 1 — что получил генератор синтетики",
              size=23, bold=True, color=COLOR_TITLE)
    _add_rect(s, 0.5, 0.88, 12.5, 0.04, fill=COLOR_ACCENT)

    _add_rect(s, 0.55, 1.05, 4.0, 5.95, fill=COLOR_HIGHLIGHT_BG)
    _add_text(s, 0.72, 1.14, 3.75, 0.35, "Конфигурация генерации", size=14, bold=True, color=COLOR_TITLE)
    manifest_block = (
        "synthetic_fewshot_manifest.json:\n"
        "{\n"
        '  "enabled": true,\n'
        '  "status": "ok",\n'
        '  "meta": {\n'
        '    "mode": "inject_as_hint",\n'
        '    "hint_in_system_message": true,\n'
        '    "n_hard_samples": 8,\n'
        '    "n_examples_target": 8,\n'
        '    "n_boundary_pairs_target": 4\n'
        "  }\n"
        "}"
    )
    _add_text(s, 0.72, 1.55, 3.65, 2.1, manifest_block, size=9, color=COLOR_BODY)
    _add_text(s, 0.72, 4.05, 3.7, 0.35, "Источники артефактов", size=13, bold=True, color=COLOR_TITLE)
    _add_bullets(s, 0.68, 4.45, 3.75, 1.55, [
        "results_all_categories_evolve_subsample/al_iter_5/",
        "synthetic_fewshot_examples.txt + manifest.json",
    ], size=9)

    _add_rect(s, 4.8, 1.05, 4.0, 5.95, fill=COLOR_WHITE, line=COLOR_SUB)
    _add_text(s, 4.95, 1.14, 3.75, 0.35,
              "Контекст из _build_messages()", size=13, bold=True, color=COLOR_TITLE)
    _add_bullets(s, 4.92, 1.55, 3.7, 4.95, [
        ("Top confusion pairs:", "какие соседние рейтинги ансамбль путает чаще всего."),
        ("Current DynamicRules:", "текущие правила промпта и границы рейтингов."),
        ("Current FewShotExamples:", "реальные few-shot, которые можно улучшить."),
        ("Hard examples:", "до 8 отзывов из hard-контекста активного батча."),
        ("Requirements:", "только синтетика, покрытие 1..5, контрастные пары, формат Rating: N."),
    ], size=10)

    _add_rect(s, 9.05, 1.05, 3.8, 5.95, fill=COLOR_YELLOW_BG, line=COLOR_ACCENT)
    _add_text(s, 9.2, 1.14, 3.55, 0.35, "Что важно показать", size=13, bold=True, color=COLOR_TITLE)
    _add_bullets(s, 9.15, 1.55, 3.55, 3.8, [
        "Синтетика не генерируется случайно.",
        "Она строится на ошибках текущего промпта и реальных hard-примерах.",
        "В режиме inject_as_hint это не жёсткая подмена: мутатор берёт только полезные фрагменты.",
    ], size=10)
    _add_text(s, 9.2, 5.75, 3.5, 0.75,
              "Дальше — не «скрин», а сами реальные примеры из raw-файлов.",
              size=11, bold=True, color=COLOR_TITLE)

    _add_footer(s, "6 / 10")


def slide_demo_example_coffee(prs: Presentation) -> None:
    s = _add_blank(prs)
    _add_text(s, 0.5, 0.25, 12.5, 0.55,
              "Демо-отчёт 2 — сгенерированная пара 5★ / 4★",
              size=23, bold=True, color=COLOR_TITLE)
    _add_rect(s, 0.5, 0.88, 12.5, 0.04, fill=COLOR_ACCENT)
    _add_text(s, 0.55, 1.05, 12.3, 0.35,
              "Источник: results_all_categories_evolve_subsample/al_iter_5/synthetic_fewshot_examples.txt",
              size=10, italic=True, color=COLOR_SUB)

    _add_rect(s, 0.6, 1.55, 6.0, 4.5, fill=COLOR_GREEN_BG, line=COLOR_SUB)
    _add_text(s, 0.8, 1.72, 5.6, 0.35, "Example 2 — Rating: 5", size=15, bold=True, color=COLOR_TITLE)
    _add_text(
        s, 0.8, 2.18, 5.55, 2.95,
        "Review: I am so happy with this coffee maker! It brews the perfect pot of coffee, "
        "it's easy to clean, and it looks great on my counter. Some people mentioned the "
        "30-minute auto-shutoff for the warmer, but I actually love that feature for safety "
        "and so the coffee doesn't get burnt. It's exactly what I wanted. Highly recommend!",
        size=13, color=COLOR_BODY,
    )
    _add_rect(s, 0.8, 5.25, 5.55, 0.55, fill=COLOR_WHITE, line=COLOR_ACCENT)
    _add_text(s, 0.95, 5.36, 5.25, 0.3,
              "Смысл: без существенных недостатков → уверенные 5★.", size=11, bold=True, color=COLOR_TITLE)

    _add_rect(s, 6.85, 1.55, 6.0, 4.5, fill=COLOR_YELLOW_BG, line=COLOR_SUB)
    _add_text(s, 7.05, 1.72, 5.6, 0.35, "Example 1 — Rating: 4", size=15, bold=True, color=COLOR_TITLE)
    _add_text(
        s, 7.05, 2.18, 5.55, 2.95,
        "Review: This coffee maker is pretty good. It brews a full pot quickly and the coffee "
        "tastes great. My only real complaint is that the warming plate only stays on for 30 minutes "
        "before shutting off, so I often come back for a second cup and find it's cold. I wish it "
        "stayed on longer, but otherwise, it's a solid machine for the price.",
        size=13, color=COLOR_BODY,
    )
    _add_rect(s, 7.05, 5.25, 5.55, 0.55, fill=COLOR_WHITE, line=COLOR_ACCENT)
    _add_text(s, 7.2, 5.36, 5.25, 0.3,
              "Смысл: хороший товар + важный, но не fatal complaint → 4★.", size=11, bold=True, color=COLOR_TITLE)
    _add_text(s, 0.65, 6.25, 12.0, 0.28,
              "Локальный эффект цикла 5: R_worst 0.560 → 0.667 (+10.7 п.п.), combined 0.844 → 0.908 (+6.4 п.п.)",
              size=10, bold=True, color=COLOR_ACCENT)

    _add_footer(s, "7 / 10")


def slide_demo_example_baking(prs: Presentation) -> None:
    s = _add_blank(prs)
    _add_text(s, 0.5, 0.25, 12.5, 0.55,
              "Демо-отчёт 3 — сгенерированная пара 3★ / 4★",
              size=23, bold=True, color=COLOR_TITLE)
    _add_rect(s, 0.5, 0.88, 12.5, 0.04, fill=COLOR_ACCENT)
    _add_text(s, 0.55, 1.05, 12.3, 0.35,
              "Источник: results_all_categories_evolve_subsample/al_iter_5/synthetic_fewshot_examples.txt",
              size=10, italic=True, color=COLOR_SUB)

    _add_rect(s, 0.6, 1.55, 6.0, 4.5, fill=COLOR_YELLOW_BG, line=COLOR_SUB)
    _add_text(s, 0.8, 1.72, 5.6, 0.35, "Example 4 — Rating: 4", size=15, bold=True, color=COLOR_TITLE)
    _add_text(
        s, 0.8, 2.18, 5.55, 2.9,
        "Review: This is a pretty good knife set for the price. The main knives feel solid, "
        "look nice, and are sharp enough for my daily cooking. The included steak knives feel "
        "a little flimsy in comparison, but that's a minor issue for me since I don't use them "
        "often. Overall, I'm pleased with the purchase.",
        size=13, color=COLOR_BODY,
    )
    _add_rect(s, 0.8, 5.25, 5.55, 0.55, fill=COLOR_WHITE, line=COLOR_ACCENT)
    _add_text(s, 0.95, 5.36, 5.25, 0.3,
              "Смысл: minor issue не ломает overall positive → 4★.", size=11, bold=True, color=COLOR_TITLE)

    _add_rect(s, 6.85, 1.55, 6.0, 4.5, fill=COLOR_RED_BG, line=COLOR_SUB)
    _add_text(s, 7.05, 1.72, 5.6, 0.35, "Example 3 — Rating: 3", size=15, bold=True, color=COLOR_TITLE)
    _add_text(
        s, 7.05, 2.18, 5.55, 2.9,
        "Review: I have mixed feelings about this knife set. The block is nice and the chef's "
        "knife and paring knife are fantastic. However, the bread knife is completely useless "
        "and just tears bread. The scissors also broke the first time I tried to cut open a plastic "
        "food package. Since some pieces are great and others are junk, the whole set is just okay.",
        size=13, color=COLOR_BODY,
    )
    _add_rect(s, 7.05, 5.25, 5.55, 0.55, fill=COLOR_WHITE, line=COLOR_ACCENT)
    _add_text(s, 7.2, 5.36, 5.25, 0.3,
              "Смысл: смешанный набор, часть компонентов junk → 3★.", size=11, bold=True, color=COLOR_TITLE)
    _add_text(s, 0.65, 6.25, 12.0, 0.28,
              "После цикла 5: Acc_Hard 0.354 → 0.397 (+4.2 п.п.) — рост именно на hard-примерах.",
              size=10, bold=True, color=COLOR_ACCENT)

    _add_footer(s, "8 / 10")


def slide_demo_example_kettle(prs: Presentation) -> None:
    s = _add_blank(prs)
    _add_text(s, 0.5, 0.25, 12.5, 0.55,
              "Демо-отчёт 4 — сгенерированная пара 2★ / 1★",
              size=23, bold=True, color=COLOR_TITLE)
    _add_rect(s, 0.5, 0.88, 12.5, 0.04, fill=COLOR_ACCENT)
    _add_text(s, 0.55, 1.05, 12.3, 0.35,
              "Источник: results_all_categories_evolve_subsample/al_iter_5/synthetic_fewshot_examples.txt",
              size=10, italic=True, color=COLOR_SUB)

    _add_rect(s, 0.6, 1.55, 6.0, 4.5, fill=COLOR_YELLOW_BG, line=COLOR_SUB)
    _add_text(s, 0.8, 1.72, 5.6, 0.35, "Rating: 2", size=15, bold=True, color=COLOR_TITLE)
    _add_text(
        s, 0.8, 2.18, 5.55, 2.9,
        "Review: I wanted to like this electric kettle because the design is very sleek. "
        "Unfortunately, it's a disappointment. It takes over 10 minutes to boil a full pot of water, "
        "which is significantly slower than my old, cheap kettle. The outside also gets extremely "
        "hot to the touch, which doesn't feel safe. It works, but it's frustrating.",
        size=13, color=COLOR_BODY,
    )
    _add_rect(s, 0.8, 5.25, 5.55, 0.55, fill=COLOR_WHITE, line=COLOR_ACCENT)
    _add_text(s, 0.95, 5.36, 5.25, 0.3,
              "Смысл: плохой опыт, но товар всё ещё функционирует → 2★.", size=11, bold=True, color=COLOR_TITLE)

    _add_rect(s, 6.85, 1.55, 6.0, 4.5, fill=COLOR_RED_BG, line=COLOR_SUB)
    _add_text(s, 7.05, 1.72, 5.6, 0.35, "Rating: 1", size=15, bold=True, color=COLOR_TITLE)
    _add_text(
        s, 7.05, 2.18, 5.55, 2.9,
        "Review: DO NOT BUY. This is a dangerous product and a fire hazard. On its second use, "
        "the kettle started making a popping noise, smelled strongly of burning plastic, and then "
        "died completely, tripping my kitchen's circuit breaker. It's a complete waste of money and "
        "I'm just glad I was home to unplug it immediately.",
        size=13, color=COLOR_BODY,
    )
    _add_rect(s, 7.05, 5.25, 5.55, 0.55, fill=COLOR_WHITE, line=COLOR_ACCENT)
    _add_text(s, 7.2, 5.36, 5.25, 0.3,
              "Смысл: опасность/непригодность → уверенный 1★.", size=11, bold=True, color=COLOR_TITLE)
    _add_text(s, 0.65, 6.25, 12.0, 0.28,
              "Важно: это cycle-level evidence — эффект цикла, где синтетика была подана мутатору.",
              size=10, bold=True, color=COLOR_ACCENT)

    _add_footer(s, "9 / 10")


def slide_demo_metrics(prs: Presentation) -> None:
    s = _add_blank(prs)
    _add_text(s, 0.5, 0.25, 12.5, 0.55,
              "Демо-отчёт 5 — какой прирост метрик это дало",
              size=23, bold=True, color=COLOR_TITLE)
    _add_rect(s, 0.5, 0.88, 12.5, 0.04, fill=COLOR_ACCENT)

    _add_text(s, 0.65, 1.08, 12.1, 0.35,
              "Главное сравнение: no-synth uncapped_train vs synth+hint evolve_subsample (все категории)",
              size=14, bold=True, color=COLOR_TITLE)
    _add_metric_box(s, 0.75, 1.65, "R_worst", "0.533 → 0.667", "+13.4 п.п. на худших пользователях", COLOR_GREEN_BG)
    _add_metric_box(s, 4.05, 1.65, "Combined score", "0.844 → 0.894", "+5.0 п.п. итоговой метрики", COLOR_HIGHLIGHT_BG)
    _add_metric_box(s, 7.35, 1.65, "Acc_Hard", "0.283 → 0.353", "+7.0 п.п. на hard-примерах", COLOR_YELLOW_BG)
    _add_metric_box(s, 10.65, 1.65, "Best prompt", "cycle = 6", "best_val_score = 0.8616", COLOR_HIGHLIGHT_BG)

    _add_rect(s, 0.75, 3.0, 3.7, 2.75, fill=COLOR_GREEN_BG, line=COLOR_ACCENT)
    _add_text(s, 0.9, 3.12, 3.45, 0.32, "Локальный эффект цикла 5", size=13, bold=True, color=COLOR_TITLE)
    _add_bullets(s, 0.85, 3.5, 3.45, 1.95, [
        "R_worst: 0.560 → 0.667 (+10.7 п.п.)",
        "combined: 0.844 → 0.908 (+6.4 п.п.)",
        "Acc_Hard: 0.354 → 0.397 (+4.2 п.п.)",
        "synthetic examples были поданы в этот AL-цикл.",
    ], size=10)

    _add_rect(s, 4.75, 3.0, 3.7, 2.75, fill=COLOR_HIGHLIGHT_BG)
    _add_text(s, 4.9, 3.12, 3.45, 0.32, "Почему это связано с демо", size=13, bold=True, color=COLOR_TITLE)
    _add_bullets(s, 4.85, 3.5, 3.45, 1.95, [
        "Генератор получает ошибки текущей модели и реальные hard-примеры.",
        "Синтетические пары показывают именно границы между соседними оценками.",
        "В режиме inject_as_hint мутатор использует примеры как материал, а не как принудительную замену.",
    ], size=10)

    _add_rect(s, 8.75, 3.0, 3.9, 2.75, fill=COLOR_YELLOW_BG)
    _add_text(s, 8.9, 3.12, 3.65, 0.32, "Честное ограничение", size=13, bold=True, color=COLOR_TITLE)
    _add_bullets(s, 8.85, 3.5, 3.65, 1.95, [
        "All-categories сравнение не полностью контролирует размер train pool.",
        "Цикл 5 — не строгая причинность одного примера: одновременно идут mutation/consolidation/refresh.",
        "Но это честная cycle-level evidence.",
    ], size=10)

    _add_rect(s, 0.75, 6.05, 11.9, 0.75, fill=COLOR_GREEN_BG, line=COLOR_ACCENT)
    _add_text(s, 0.95, 6.17, 11.5, 0.45,
              "Вывод: прирост даёт не «длинный промпт», а явные граничные few-shot демонстрации, "
              "построенные из ошибок текущего промпта.",
              size=12, bold=True, color=COLOR_TITLE)

    _add_footer(s, "10 / 10")


def main() -> None:
    prs = _new_prs()
    slide_title(prs)
    slide_intro(prs)
    slide_article_overview(prs)
    slide_method(prs)
    slide_results(prs)
    slide_demo_input_report(prs)
    slide_demo_example_coffee(prs)
    slide_demo_example_baking(prs)
    slide_demo_example_kettle(prs)
    slide_demo_metrics(prs)
    prs.save(OUT)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
