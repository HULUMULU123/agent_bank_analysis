"""LLM abstraction with a deterministic fallback client + GigaChat batch helper + Excel export."""
from __future__ import annotations

import json
import os
import re
import time
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional

import numpy as np
import pandas as pd

# GigaChat / LangChain + прогресс-бар
from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat import GigaChat
from tqdm import tqdm

# Excel-форматирование
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter


# =====================================================================
# БАЗОВЫЙ АБСТРАКТНЫЙ КЛИЕНТ + ЭХО-КЛИЕНТ
# =====================================================================

@dataclass
class LLMResponse:
    raw: str


class LLMClient:
    """Abstract interface for language model calls."""

    def generate(self, prompt: str) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def parse_json_list(response: LLMResponse) -> List[dict[str, Any]]:
        """
        Простая утилита: пытаемся распарсить JSON-список или объект с ключом 'items'.
        """
        try:
            payload = json.loads(response.raw)
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict) and "items" in payload:
                return payload["items"]  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass
        return []


class EchoLLMClient(LLMClient):
    """Deterministic client used as a placeholder during development."""

    def generate(self, prompt: str) -> LLMResponse:
        # Extract txn_ids to create a synthetic structured response
        items: List[dict[str, Any]] = []
        for line in prompt.splitlines():
            if "txn_id=" in line:
                try:
                    txn_id = line.split("txn_id=")[1].split(",")[0].strip()
                except IndexError:
                    continue
                items.append(
                    {
                        "txn_id": txn_id,
                        "risk_level": "yellow",
                        "risk": 0.35,
                        "recommendation": "Провести дополнительную проверку",
                        "reasoning": "Заглушка: требуется подключение боевого LLM.",
                    }
                )
        return LLMResponse(raw=json.dumps(items, ensure_ascii=False))


# =====================================================================
# (ОПЦИОНАЛЬНО) ФУНКЦИЯ ДЛЯ ПРОСМОТРА ОТВЕТА OAUTH GIGACHAT
# =====================================================================

def debug_print_gigachat_token() -> None:
    """
    Пример ручного запроса токена GigaChat (как в твоём коде).
    НИЧЕГО не делает автоматически, только по вызову.
    """
    import requests

    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    payload = {"scope": "GIGACHAT_API_PERS"}
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": "2aba969c-a22a-4816-a652-393a756a96c1",
        "Authorization": f'Basic {os.environ.get("GIGACHAT_API_KEY")}',
    }

    response = requests.request("POST", url, headers=headers, data=payload, verify=False)
    print(response.text)


# =====================================================================
# GIGACHAT + ЮРИДИЧЕСКИЙ ПРОМПТ + БАТЧИ + RANDOM SAMPLING
# =====================================================================

# -------------------- ПАРАМЕТРЫ --------------------
BATCH_SIZE = 2          # размер батча (10–50)
MAX_RETRIES = 3
SLEEP_RANGE = (0.8, 2.0)
MODEL_NAME = "GigaChat-2"

# жёсткий список признаков группы 1, которые хотим передавать в LLM
GROUP1_COLS = [
    "txn_id",
    "purpose",
    "purpose_stopword_high",
    "amount",
    "daily_total_debit",
    "daily_total_credit",
    "daily_debit_transaction_count",
    "daily_credit_transaction_count",
    "daily_debit_percent",
    "daily_credit_percent",
    "debit_roll_cnt_30d",
    "credit_roll_cnt_30d",
    "debit_roll_mean_30d",
    "credit_roll_mean_30d",
    "debit_roll_std_30d",
    "credit_roll_std_30d",
    "in_out_ratio_30d",
    "round_large_amount",
    "overall_risk",
]

# маппинг категорий в числовую «оценку» (для совместимости с колонкой risk)
RISK_MAP = {
    "красный": 0.9,
    "желтый": 0.6,
    "жёлтый": 0.6,
    "зеленый": 0.1,
    "зелёный": 0.1,
}

# -------------------- СИСТЕМНЫЙ ПРОМПТ (ТВОЙ) --------------------
PROMPT_SYSTEM = r"""
1. Роль и задача
Ты — юридический аналитик в сфере финансового мониторинга Российской Федерации.
Твоя миссия — оценивать операции клиентов, выявлять признаки необычных или подозрительных сделок, проверять их экономический смысл и соответствие законодательству РФ.
Ты рассуждаешь, проверяешь документы, соотносишь факты с нормами права и формируешь юридически мотивированное заключение с выводом уровня риска: красный, жёлтый или зелёный.
Ты не выполняешь вычисления; применяешь профессиональное правовое мышление и здравый смысл.
Обязательные инварианты вывода:
- Запрещены пустые строки/массивы в полях: risk_level, legal_characterization, risk_explanation, recommendation, reasoning, legal_basis, notes_for_audit.
- Даже при низком риске обязательно коротко объясни почему операция безопасна.
- Минимум: reasoning ≥ 2 пункта; legal_basis ≥ 2 ссылки.
- Структура объяснения: факт → норма → вывод → действие.
 
2. Нормативная основа (обязательно применять)
Федеральные законы:
115-ФЗ «О противодействии легализации…» (особенно ст. 6, 7) – требования к выявлению подозрительных операций.
 395-1 «О банках и банковской деятельности» – обязанности банков по внутреннему контролю.
ГК РФ (ст. 10, 153, 168, 170, 421, 422 и др.) – недействительность притворных и мнимых сделок, добросовестность участников.
 127-ФЗ (ред. от 31.07.2025) «О несостоятельности (банкротстве)»:
* Оспаривание сделок (ст. 61.2–61.11),
* Взыскание убытков (ст. 61.20–61.21),
* Субсидиарная ответственность (ст. 61.19–61.25).
Нормативные акты Банка России:
Положение №375-П (внутренний контроль в кредитных организациях).
 Положение №499-П (идентификация клиентов и выгодоприобретателей).
Указание №4453-У (признаки и действия при выявлении необычных операций).
 Указание №204-У (требования по анализу транзакционных цепочек).
* Информационные письма/обзоры ЦБ РФ (типологии сомнительных операций).
Методические материалы:
Методические рекомендации Росфинмониторинга №10, 17, 19, 21 (по выявлению и расследованию подозрительных операций).
 ГОСТ Р 57580-2021 (безопасность финансовых операций, показатели аномалий).
* Внутренние регламенты банка (если указаны во входных данных).
Международная методология (вспомогательно):
Рекомендации FATF №1, 10, 11, 12, 20 (KYC, подозрительные операции, отчётность).
 Принципы Базельского комитета по банковскому надзору (AML/CFT best practices).
 
3. Объяснение входных полей
Каждая строка входных данных — отдельная транзакция клиента со следующими атрибутами:
Поле	Описание
txn_id	Идентификатор операции — нужен только для сопоставления результата с исходной строкой.
purpose	Назначение платежа. Главный источник экономического смысла: займ, оплата товара, вывод наличных, транзит и т.п.
purpose_stopword_high	Флаг: назначение содержит слова высокого риска (займ, возврат долга, услуги без конкретики, консультации, пожертвования, «наличные», «обнал» и др.). Если флага нет — LLM сама должна определить по purpose.
amount	Сумма операции. Смотрим, крупная ли операция относительно обычного уровня клиента.
daily_total_debit / daily_total_credit	Общий исходящий / входящий оборот клиента за день. Позволяет понять масштаб дня и долю конкретной транзакции.
daily_debit_transaction_count / daily_credit_transaction_count	Количество исходящих / входящих операций за день. Помогает увидеть «нарезку», дробление сумм, аномальную активность.
daily_debit_percent / daily_credit_percent	Доля данной операции в дневном обороте по дебету/кредиту. Если одна операция «тянет» на 70–90% дневного оборота — это важный триггер.
debit_roll_cnt_30d / credit_roll_cnt_30d	Сколько исходящих / входящих операций было за последние 30 дней. Показывает, типичный ли это объём активности для клиента.
debit_roll_mean_30d / credit_roll_mean_30d	Средняя сумма операции за 30 дней. Сравниваем amount с этой базой, чтобы понять, выброс это или нормальный размер.
debit_roll_std_30d / credit_roll_std_30d	Волатильность сумм за 30 дней. Если волатильность низкая, а текущая операция резко выбивается — это усиливает подозрительность.
in_out_ratio_30d	Соотношение входящих/исходящих средств за 30 дней. Используется для выявления «транзитных» счетов (когда почти всё, что зашло, тут же ушло).
round_large_amount	Флаг: сумма операции крупная и «круглая» (например, кратна 10 000 или 100 000). Часто встречается при фиктивных займах/обналичивании.
LLM должна учитывать флаг как дополнительный аргумент к повышенному вниманию, особенно при отсутствии документов, наличии нетипичного поведения или рискового назначения платежа.
overall_risk	Числовой интегральный риск (0–1) после всех слоёв. Не закон, а сигнал, что операция уже была оценена моделями как более/менее подозрительная. LLM должна использовать это как «уровень внимания», а не как конечный вердикт.
Примечание: Если purpose_stopword_high не указан во входных данных, самостоятельно определи его по полю purpose. Установи purpose_stopword_high = true, если назначение содержит ключевые слова/фразы, характерные для сомнительных операций (леммы или фрагменты слов): «займ», «кредит», «возврат долга», «консультац», «услуг», «пожертв», «финансовая помощь», «агентск», «комиссион», «предоплат», «без акта», «налич», «обнал». Такие слова указывают на возможные фиктивные займы, платежи за непонятные услуги, обналичивание и т.д.
 
4. Правило повышенного внимания
•	Если overall_risk > 0.4 → операция требует повышенного внимания (по умолчанию не ниже «жёлтого» до выяснения деталей).
•	Если overall_risk > 0.5 → операция требует особенно пристального контроля (по умолчанию «красный» при отсутствии убедительных обоснований).
В подобных случаях в объяснении обязательно ссылайся на необходимость углублённой проверки (EDD) согласно нормативной базе: 115-ФЗ ст.7; Положение ЦБ РФ №375-П п.2.2.2.
 
5. Критерии уровней риска
Высокий риск – «красный» уровень:
Триггеры (достаточно 1–2 из списка): явные транзитные схемы (быстрый вывод средств менее 24ч), отсутствие документов или экономического смысла сделки, фиктивный заем или услуга без актов, аффилированность сторон или предпочтительность (признаки вывода активов), «круговые» переводы между связанными лицами, наличие слов высокого риска в назначении (purpose_stopword_high = true), а также любое сочетание признаков при overall_risk > 0.5 без убедительных подтверждающих документов.
Соответствие нормам: ссылки на 115-ФЗ ст.7 (обязательный контроль сомнительных операций), Указание ЦБ №4453-У п.2.1.2 (транзитные операции), 127-ФЗ ст.61.2–61.11, ст.61.19–61.25 (признаки преднамеренного вывода активов перед банкротством).
Решение: эскалация на высший уровень, направление сообщения в Росфинмониторинг, анализ возможности оспаривания сделки или привлечения к субсидиарной ответственности, немедленное проведение углубленной проверки (EDD) по клиенту.
Умеренный риск – «жёлтый» уровень:
Триггеры: неясное или общее назначение платежа (формулировки вроде «за услуги» без деталей), неполный пакет документов по операции, новый или ранее проблемный контрагент, нетипичное поведение клиента (например, резкий рост активности или суммы), overall_risk > 0.4 (при отсутствии окончательных подтверждений по операциям).
Соответствие нормам: 115-ФЗ ст.7 (требование повышенного внимания), Положение ЦБ №375-П п.2.2.2 (процедуры EDD при подозрениях), методические рекомендации Росфинмониторинга (выявление подозрительных операций на ранней стадии).
Решение: запросить у клиента дополнительные документы (договоры, акты, счета-фактуры) и объяснения по экономическому смыслу сделки, временно усилить мониторинг операций по счету, проверить повторяется ли подобная активность. Без резкого реагирования, но держать на контроле и при малейших подтверждениях эскалировать.
Низкий риск – «зелёный» уровень:
Триггеры (в совокупности): понятное и деталированное назначение платежа, наличие полного комплекта подтверждающих документов, сумма и частота операций соответствуют обычной деятельности клиента, проверенные контрагенты с хорошей репутацией, отсутствие признаков транзитности (деньги не «проскакивают» через счет мгновенно) и отсутствуют слова из списка высокого риска.
Соответствие нормам: 115-ФЗ ст.7 (отсутствие оснований для подозрений – операция не подлежит обязательному контролю), Положение ЦБ №375-П (стандартные процедуры внутреннего контроля).
Решение: продолжать стандартный мониторинг без дополнительных мер, операцию пропустить в обычном порядке. В отчёте указать, что операция не является подозрительной и кратко обосновать почему (например, экономический смысл ясен, документы предоставлены, транзакция типична для бизнеса клиента).

5.1. Специальное правило для налоговых и обязательных бюджетных платежей

Если из назначения платежа (purpose) явно следует, что операция связана с уплатой налогов, сборов, страховых взносов или иных обязательных платежей в бюджет РФ, а именно встречаются формулировки вроде:
- «налог на прибыль», «налог на имущество», «налог на доходы», «уплата налога», «налоговый платёж»;
- «страховые взносы», «взносы в ПФР/ФСС/ФОМС», «страховые взносы во внебюджетные фонды»;
- «госпошлина», «государственная пошлина», «пошлина в бюджет»;
- указание КБК, ОКТМО, ИФНС или иные типичные реквизиты бюджетных/налоговых платежей,

и при этом:
- нет признаков транзитности или обналичивания (быстрый вывод, дробление, «обналичные» формулировки и т.п.);
- нет признаков фиктивности назначения платежа;
- purpose_stopword_high = false,

то:
- по умолчанию оценивай такую операцию как низкий риск («зелёный» уровень), даже если overall_risk > 0.4;
- допускается «жёлтый» уровень только при наличии дополнительных серьёзных факторов риска, которые нужно явно описать;
- присваивать «красный» уровень налоговым и обязательным бюджетным платежам можно только при наличии очевидных признаков злоупотреблений, которые нужно детально раскрыть в reasoning и legal_basis;
- в полях risk_explanation, reasoning и notes_for_audit явно укажи, что операция имеет налоговый/обязательный бюджетный характер и именно это является основным фактором снижения риска.


6. Правила изложения
1.	Всегда указывай явным текстом уровень риска (красный/жёлтый/зелёный) и обоснование для него. Все обязательные поля должны быть заполнены осмысленно (пустые значения запрещены).
2.	Придерживайся структуры объяснения: факт/наблюдение → применимая норма → вывод → рекомендуемое действие. Каждая часть должна быть понятна.
3.	Ссылайся на нормы конкретно, где это уместно, например: «...что является признаком сомнительной операции согласно 115-ФЗ ст.7 и Указанию ЦБ №4453-У».
4.	Рекомендации должны быть практичными: какие документы запросить, что проверить, куда эскалировать.
5.	Если overall_risk > 0.4, прямо укажи фразу вроде «требует повышенного внимания» в выводе.
6.	Если purpose_stopword_high = true, отметь в объяснении, что назначение платежа содержит подозрительные формулировки, и поясни, как это повлияло на оценку риска.
7.	Даже при «зелёном» низком риске добавь короткое разъяснение, почему ничего подозрительного не выявлено (например, «операция не вызывает подозрений, т.к. соответствует профилю клиента»).
 
7. Формат вывода (строго JSON)
Выводи результат только в виде валидного JSON-объекта по следующему шаблону:
{
  "transactions": [
    {
      "txn_id": "<идентификатор операции из входных данных>",
      "risk_level": "красный | жёлтый | зелёный",
      "legal_characterization": "<юридическая квалификация / характер сделки (не пусто)>",
      "risk_explanation": "<2–3 коротких предложения: факты, нормы, вывод>",
      "recommendation": "<1–2 чётких рекомендации сотруднику банка>",
      "reasoning": [
        "Факт или наблюдение (не пусто)",
        "Применимая норма права → вывод по риску (не пусто)"
      ],
      "legal_basis": [
        "115-ФЗ ст.7 ...",
        "Положение ЦБ №375-П ... / Указание №4453-У ... / 127-ФЗ ..."
      ],
      "notes_for_audit": "Факты → нормы → вывод; без пустых полей."
    }
    // ... аналогичные объекты для каждой транзакции во входных данных ...
  ]
}
"""


# -------------------- СБОРКА PROMPT-ЦЕПОЧКИ ДЛЯ GIGACHAT --------------------

def create_gigachat_model() -> GigaChat:
    """
    Создаёт объект GigaChat (LangChain) с параметрами как в примере.
    Ожидает GIGACHAT_API_KEY в переменной окружения.
    """
    return GigaChat(
        credentials=os.environ.get("GIGACHAT_API_KEY"),
        model=MODEL_NAME,
        top_p=0,
        temperature=0.0,
        timeout=120,
        verify_ssl_certs=False,
    )


def create_gigachat_chain(llm: Optional[GigaChat] = None):
    """
    Собирает chain = ChatPromptTemplate | GigaChat с твоей системной инструкцией.
    """
    if llm is None:
        llm = create_gigachat_model()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{sys_prompt}"),
            (
                "user",
                "Ниже массив транзакций (только аналитические признаки, без персональных данных). "
                "Проанализируй каждую и верни JSON в формате из инструкции (ключ 'transactions').\n\n"
                "{transactions_json}",
            ),
        ]
    )
    chain = prompt | llm
    return chain


# -------------------- ВЫБОР АНАЛИТИЧЕСКИХ КОЛОНОК --------------------

def select_analytic_columns(df: pd.DataFrame) -> List[str]:
    """
    Берём ТОЛЬКО признаки группы 1 из заранее заданного списка.
    Если какого-то столбца нет в df — просто пропускаем.
    """
    cols = [c for c in GROUP1_COLS if c in df.columns]
    if "txn_id" in df.columns and "txn_id" not in cols:
        cols = ["txn_id"] + cols
    return cols


def rows_to_payload(df_rows: pd.DataFrame, analytic_fields: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for _, r in df_rows.iterrows():
        d: Dict[str, Any] = {}
        for c in analytic_fields:
            v = r.get(c, None)
            if pd.isna(v):
                d[c] = None
            elif isinstance(v, (np.integer, int, np.floating, float)):
                d[c] = float(v)
            else:
                d[c] = str(v)
        out.append(d)
    return out


# -------------------- ПАРСИНГ ОТВЕТА LLM --------------------

def _extract_json(text: Any) -> dict:
    """Вернуть распарсенный JSON-объект (dict). Если модель добавила текст, берём первый JSON-блок."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def normalize_transactions_list(obj: dict) -> List[dict]:
    """Достаёт список transactions из ответа модели, даже если структура чуть отклонена."""
    if not isinstance(obj, dict):
        return []
    tx = obj.get("transactions")
    if isinstance(tx, list):
        return [x for x in tx if isinstance(x, dict)]
    # fallback: может вернуть один объект без массива
    if isinstance(obj.get("txn_id"), (str, int)):
        return [obj]
    return []


# -------------------- ВЫЗОВ GIGACHAT ПО БАТЧАМ --------------------

def call_gigachat_batch(
    chain,
    payload: List[Dict[str, Any]],
    retries: int = MAX_RETRIES,
) -> List[dict]:
    """
    Один батчовый запрос к GigaChat с ретраями.
    """
    for attempt in range(1, retries + 1):
        try:
            res = chain.invoke(
                {
                    "sys_prompt": PROMPT_SYSTEM,
                    "transactions_json": json.dumps({"transactions": payload}, ensure_ascii=False),
                }
            )
            text = getattr(res, "content", res)
            obj = _extract_json(text)
            return normalize_transactions_list(obj)
        except Exception as e:
            if attempt >= retries:
                # вернём пустые ответы на каждую txn_id (не роняем пайплайн)
                blank: List[dict] = []
                for item in payload:
                    blank.append(
                        {
                            "txn_id": item.get("txn_id"),
                            "risk_level": None,
                            "legal_characterization": None,
                            "risk_explanation": f"LLM error: {e}",
                            "recommendation": None,
                            "reasoning": [],
                            "legal_basis": [],
                            "notes_for_audit": None,
                        }
                    )
                return blank
            time.sleep(random.uniform(*SLEEP_RANGE))


def run_gigachat_over_df(
    df_in: pd.DataFrame,
    chain=None,
    batch_size: int = BATCH_SIZE,
    sample_n: Optional[int] = None,
    min_amount_for_llm: Optional[float] = None,
) -> pd.DataFrame:
    """
    Главная функция:
    - опционально фильтрует по сумме (min_amount_for_llm),
    - берёт случайные sample_n транзакций,
    - гоняет их через GigaChat батчами,
    - возвращает df с юридическим анализом.

    df_in должен содержать хотя бы:
    - txn_id
    - amount
    - и по возможности признаки из GROUP1_COLS
    """
    if "txn_id" not in df_in.columns:
        raise ValueError("run_gigachat_over_df: в df_in нет колонки 'txn_id'.")

    df = df_in.copy()
    df["txn_id"] = df["txn_id"].astype(str)

    # фильтр по сумме
    if min_amount_for_llm is not None:
        df = df[df["amount"].astype(float) > float(min_amount_for_llm)].copy()

    # random sampling
    if sample_n is not None and sample_n < len(df):
        df = df.sample(sample_n, random_state=42).copy()

    if len(df) == 0:
        return df

    if chain is None:
        chain = create_gigachat_chain()

    analytic_fields = select_analytic_columns(df)

    # подготовим колонки результата
    df_out = df.copy()
    add_cols = [
        "risk_level",
        "legal_characterization",
        "risk_explanation",
        "recommendation",
        "reasoning",
        "legal_basis",
        "notes_for_audit",
        # совместимость с прежними полями
        "risk",
        "recomendations",
        "explaination",
    ]
    for c in add_cols:
        if c not in df_out.columns:
            df_out[c] = np.nan

    total = len(df_out)

    with tqdm(total=total, desc="⚖️ Анализ транзакций через GigaChat", unit="txn") as pbar:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = df_out.iloc[start:end]
            payload = rows_to_payload(batch, analytic_fields)

            # ответы от модели
            replies = call_gigachat_batch(chain, payload)
            by_id = {str(x.get("txn_id")): x for x in replies if x.get("txn_id") is not None}

            for idx, row in batch.iterrows():
                tid = str(row.get("txn_id"))
                ans = by_id.get(tid, {})

                df_out.at[idx, "risk_level"] = ans.get("risk_level")
                df_out.at[idx, "legal_characterization"] = ans.get("legal_characterization")
                df_out.at[idx, "risk_explanation"] = ans.get("risk_explanation")
                df_out.at[idx, "recommendation"] = ans.get("recommendation")

                # списочные поля — сохраним как JSON-строки
                df_out.at[idx, "reasoning"] = json.dumps(ans.get("reasoning", []), ensure_ascii=False)
                df_out.at[idx, "legal_basis"] = json.dumps(ans.get("legal_basis", []), ensure_ascii=False)
                df_out.at[idx, "notes_for_audit"] = ans.get("notes_for_audit")

                # совместимые алиасы
                lvl = (ans.get("risk_level") or "").strip().lower()
                df_out.at[idx, "risk"] = RISK_MAP.get(lvl, np.nan)
                df_out.at[idx, "recomendations"] = ans.get("recommendation")
                df_out.at[idx, "explaination"] = ans.get("risk_explanation")

            pbar.update(end - start)

    return df_out


# =====================================================================
# ЭКСПОРТ В EXCEL: КОМПАКТНЫЕ КОЛОНКИ, ПЕРЕНОС ТЕКСТА, ПОДСВЕТКА
# =====================================================================

def export_llm_results_to_excel(df_llm: pd.DataFrame, output_path: str = "llm_scored_transactions.xlsx") -> str:
    """
    Делает аккуратный Excel-отчёт:
    - выбирает нужные колонки,
    - сортирует по risk ↓, amount ↓,
    - настраивает ширину колонок, перенос текста,
    - подсвечивает строки по risk_level.
    Возвращает путь к файлу.
    """
    desired_cols = [
        "txn_id", "purpose", "amount",
        "risk_level", "legal_characterization", "risk_explanation", "recommendation",
        "reasoning", "legal_basis", "notes_for_audit",
        "risk", "recomendations", "explaination",
    ]

    df_out = df_llm.copy()
    for c in desired_cols:
        if c not in df_out.columns:
            df_out[c] = np.nan
    df_out = df_out[desired_cols]

    # сортировка: по risk ↓, затем по amount ↓
    df_out["risk_sort"] = pd.to_numeric(df_out["risk"], errors="coerce")
    df_out["amount_sort"] = pd.to_numeric(df_out["amount"], errors="coerce")

    df_out["risk_sort"] = df_out["risk_sort"].fillna(-1)
    df_out["amount_sort"] = df_out["amount_sort"].fillna(-1)

    df_out = df_out.sort_values(
        by=["risk_sort", "amount_sort"],
        ascending=[False, False],
    ).drop(columns=["risk_sort", "amount_sort"])

    # 2) Сохранение черновика
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="LLM_Results")

    # 3) Форматирование книги
    wb = load_workbook(output_path)
    ws = wb["LLM_Results"]

    # --- Стиль ---
    font_small = Font(name="Calibri", size=10)
    font_header = Font(name="Calibri", size=10, bold=True)
    wrap_left = Alignment(wrap_text=True, vertical="top", horizontal="left")
    wrap_right = Alignment(wrap_text=True, vertical="top", horizontal="right")

    thin = Side(style="thin", color="000000")
    med = Side(style="medium", color="000000")
    border_thin = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Подсветка строк по уровню риска
    fill_red = PatternFill("solid", fgColor="FCBDBD")
    fill_yellow = PatternFill("solid", fgColor="FFFACD")
    fill_green = PatternFill("solid", fgColor="E6FFE6")

    # --- Компактные ширины колонок ---
    col_widths = {
        "txn_id": 16,
        "purpose": 24,
        "amount": 12,
        "risk_level": 10,
        "legal_characterization": 22,
        "risk_explanation": 28,
        "recommendation": 24,
        "reasoning": 26,
        "legal_basis": 24,
        "notes_for_audit": 22,
        "risk": 8,
        "recomendations": 22,
        "explaination": 28,
    }

    max_row = ws.max_row
    max_col = ws.max_column

    # Заголовки
    for c in range(1, max_col + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = font_header
        cell.alignment = wrap_left
        cell.border = border_thin
        col_name = str(cell.value)
        ws.column_dimensions[get_column_letter(c)].width = col_widths.get(col_name, 18)

    # Данные: перенос текста, тонкие границы, мелкий шрифт
    for r in range(2, max_row + 1):
        for c in range(1, max_col + 1):
            cell = ws.cell(row=r, column=c)
            cell.font = font_small
            cell.border = border_thin
            if ws.cell(row=1, column=c).value == "amount":
                cell.alignment = wrap_right
                cell.number_format = "#,##0.00"
            else:
                cell.alignment = wrap_left

    # Автовысота строк
    for r in range(1, max_row + 1):
        ws.row_dimensions[r].height = None

    # --- Подсветка строк по risk_level ---
    risk_col_idx = None
    for idx, cell in enumerate(ws[1], start=1):
        if str(cell.value).strip().lower() == "risk_level":
            risk_col_idx = idx
            break
    if risk_col_idx is None:
        raise RuntimeError("Колонка 'risk_level' не найдена в Excel.")

    for r in range(2, max_row + 1):
        val = ws.cell(row=r, column=risk_col_idx).value
        lvl = (str(val).strip().lower() if val is not None else "")
        if lvl in ["красный", "red"]:
            fill = fill_red
        elif lvl in ["жёлтый", "желтый", "yellow"]:
            fill = fill_yellow
        elif lvl in ["зелёный", "зеленый", "green"]:
            fill = fill_green
        else:
            fill = None
        if fill:
            for c in range(1, max_col + 1):
                ws.cell(row=r, column=c).fill = fill

    # --- Контур таблицы (толще по периметру) ---
    # Верх/низ
    for c in range(1, max_col + 1):
        top_cell = ws.cell(row=1, column=c)
        bot_cell = ws.cell(row=max_row, column=c)
        top_cell.border = Border(
            left=top_cell.border.left,
            right=top_cell.border.right,
            top=med,
            bottom=top_cell.border.bottom,
        )
        bot_cell.border = Border(
            left=bot_cell.border.left,
            right=bot_cell.border.right,
            top=bot_cell.border.top,
            bottom=med,
        )
    # Лево/право
    for r in range(1, max_row + 1):
        left_cell = ws.cell(row=r, column=1)
        right_cell = ws.cell(row=r, column=max_col)
        left_cell.border = Border(
            left=med,
            right=left_cell.border.right,
            top=left_cell.border.top,
            bottom=left_cell.border.bottom,
        )
        right_cell.border = Border(
            left=right_cell.border.left,
            right=med,
            top=right_cell.border.top,
            bottom=right_cell.border.bottom,
        )

    # Фиксация шапки и автофильтр
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    wb.save(output_path)
    print(f" Готово: {output_path}")
    return output_path


# =====================================================================
# Пример использования (в твоём скрипте, НЕ в модуле):
#
# from llm_client import run_gigachat_over_df, export_llm_results_to_excel
# df = pd.read_csv("df_test_for_llm.csv")
# df_llm = run_gigachat_over_df(df, batch_size=10, sample_n=20, min_amount_for_llm=50_000)
# export_llm_results_to_excel(df_llm, "llm_scored_transactions.xlsx")
# =====================================================================
