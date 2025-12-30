import json
import os
import random
from typing import Dict, List, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------
# Banks
# -----------------------

EN_CITIES = ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa", "Edmonton", "Winnipeg", "Quebec City"]
EN_ORGS = ["Northlake Lab", "Brightwell Institute", "Redwood Analytics", "Cinder Robotics", "Maple Systems"]
EN_PRODUCTS = ["Helios", "Nimbus", "Aster", "Kite", "Juniper", "Saffron", "Raven"]
EN_TOPICS = ["astronomy", "biology", "cryptography", "robotics", "linguistics", "economics", "geology", "medicine"]

FR_CITIES = ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Strasbourg", "Bordeaux"]
FR_ORGS = ["Laboratoire Northlake", "Institut Brightwell", "Analyse Redwood", "Robotique Cinder", "Systèmes Érable"]
FR_PRODUCTS = ["Hélios", "Nimbus", "Astre", "Cerf-volant", "Genévrier", "Safran", "Corbeau"]
FR_TOPICS = ["astronomie", "biologie", "cryptographie", "robotique", "linguistique", "économie", "géologie", "médecine"]

ZH_CITIES = ["北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "西安"]
ZH_ORGS = ["北湖实验室", "明朗研究院", "红杉数据组", "灰烬机器人团队", "枫叶系统公司"]
ZH_PRODUCTS = ["赫利俄斯", "雨云", "星芒", "风筝", "杜松", "藏红花", "渡鸦"]
ZH_TOPICS = ["天文学", "生物学", "密码学", "机器人学", "语言学", "经济学", "地质学", "医学"]


def rand_year(rng: random.Random) -> int:
    return rng.randint(1998, 2025)


def rand_id(rng: random.Random) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(rng.choice(alphabet) for _ in range(8))


# -----------------------
# Metric/value cleanup
# -----------------------

# Canonical metric families for value formatting:
# - latency -> ms
# - throughput -> req/s
# - accuracy -> %
# - uptime -> %
EN_METRICS = {
    "latency": "latency",
    "throughput": "throughput",
    "accuracy": "accuracy",
    "recall": "accuracy",   # treat as percentage for formatting
    "cost": "cost",         # $/1k requests style
    "uptime": "uptime",
}

FR_METRICS = {
    "latence": "latency",
    "débit": "throughput",
    "précision": "accuracy",
    "rappel": "accuracy",
    "coût": "cost",
    "disponibilité": "uptime",
}

ZH_METRICS = {
    "延迟": "latency",
    "吞吐量": "throughput",
    "准确率": "accuracy",
    "召回率": "accuracy",
    "成本": "cost",
    "可用性": "uptime",
}


def metric_value(rng: random.Random, family: str) -> str:
    if family == "latency":
        return f"{rng.randint(8, 450)}ms"
    if family == "throughput":
        return f"{rng.randint(10, 600)} req/s"
    if family == "accuracy":
        return f"{rng.randint(60, 99)}%"
    if family == "uptime":
        return f"{rng.randint(95, 100)}%"
    # cost
    dollars = rng.choice([0.05, 0.08, 0.10, 0.12, 0.15, 0.20])
    return f"${dollars:.2f}/1k"


# -----------------------
# Templates
# -----------------------

def en_templates(f: Dict) -> List[str]:
    return [
        f"In {f['year']}, {f['org']} in {f['city']} released the {f['product']} system for {f['topic']}. "
        f"The internal ticket is {f['id']}. The benchmark reports {f['metric']} = {f['value']} and score {f['score']}/100.",
        f"{f['org']} operates out of {f['city']}. Their {f['topic']} team started the {f['product']} project in {f['year']}. "
        f"It is tracked as {f['id']}. The most recent evaluation score is {f['score']} out of 100.",
        f"Product: {f['product']}. Owner: {f['org']} ({f['city']}). Domain: {f['topic']}. Start year: {f['year']}. "
        f"ID: {f['id']}. Key metric: {f['metric']} ({f['value']}). Score: {f['score']}/100.",
        f"Meeting notes — {f['org']}, {f['city']}: The {f['product']} effort (topic: {f['topic']}) began in {f['year']}. "
        f"Reference {f['id']}. Team reported score {f['score']}/100; {f['metric']} measured at {f['value']}.",
        f"Incident recap for {f['product']}: {f['org']} ({f['city']}) confirmed the project start year as {f['year']}. "
        f"Case {f['id']}. Post-incident score: {f['score']}/100. Reported {f['metric']}: {f['value']}.",
    ]


def fr_templates(f: Dict) -> List[str]:
    return [
        f"En {f['year']}, {f['org']} à {f['city']} a lancé le système {f['product']} pour la {f['topic']}. "
        f"L'identifiant interne est {f['id']}. Rapport: {f['metric']} = {f['value']} et score {f['score']}/100.",
        f"{f['org']} est basé à {f['city']}. Son équipe de {f['topic']} a démarré le projet {f['product']} en {f['year']}. "
        f"Référence {f['id']}. Score le plus récent : {f['score']} sur 100.",
        f"Produit : {f['product']}. Propriétaire : {f['org']} ({f['city']}). Domaine : {f['topic']}. Année de début : {f['year']}. "
        f"ID : {f['id']}. Métrique clé : {f['metric']} ({f['value']}). Score : {f['score']}/100.",
        f"Notes de réunion — {f['org']}, {f['city']} : le projet {f['product']} (thème : {f['topic']}) a commencé en {f['year']}. "
        f"Référence {f['id']}. Score {f['score']}/100 ; {f['metric']} = {f['value']}.",
        f"Récapitulatif d'incident pour {f['product']} : {f['org']} ({f['city']}) confirme l'année de début {f['year']}. "
        f"Dossier {f['id']}. Score : {f['score']}/100. {f['metric']} : {f['value']}.",
    ]


def zh_templates(f: Dict) -> List[str]:
    return [
        f"{f['year']}年，{f['org']}在{f['city']}推出了{f['product']}系统，研究方向为{f['topic']}。内部编号是{f['id']}。"
        f"最新报告显示{f['metric']}={f['value']}，综合得分{f['score']}/100。",
        f"{f['org']}位于{f['city']}。其{f['topic']}团队在{f['year']}年启动{f['product']}项目，编号{f['id']}。"
        f"最近一次评估得分为{f['score']}/100。",
        f"产品：{f['product']}。机构：{f['org']}（{f['city']}）。领域：{f['topic']}。启动年份：{f['year']}。编号：{f['id']}。"
        f"关键指标：{f['metric']}（{f['value']}）。得分：{f['score']}/100。",
        f"会议纪要—{f['org']}（{f['city']}）：{f['product']}项目（{f['topic']}）于{f['year']}年开始，参考编号{f['id']}。"
        f"团队报告得分{f['score']}/100，{f['metric']}为{f['value']}。",
        f"{f['product']}事件复盘：{f['org']}（{f['city']}）确认项目启动年份为{f['year']}，工单{f['id']}。"
        f"复盘后得分{f['score']}/100，{f['metric']}={f['value']}。",
    ]


# -----------------------
# Questions (vary style; include product to disambiguate across org)
# -----------------------

def en_questions(doc_id: str, f: Dict, rng: random.Random) -> List[Dict]:
    q_year = rng.choice([
        f"In what year did {f['org']} begin {f['product']}?",
        f"What is the start year for {f['product']} at {f['org']}?",
        f"When did the {f['product']} project at {f['org']} start?",
    ])
    q_city = rng.choice([
        f"In which city is the {f['product']} team at {f['org']} based?",
        f"Where is {f['org']} located for the {f['product']} project?",
        f"Which city is associated with {f['org']}'s {f['product']} project?",
    ])
    return [
        {"qid": f"{doc_id}_q1", "question": q_year, "answer": str(f["year"])},
        {"qid": f"{doc_id}_q2", "question": q_city, "answer": f["city"]},
    ]


def fr_questions(doc_id: str, f: Dict, rng: random.Random) -> List[Dict]:
    q_year = rng.choice([
        f"En quelle année {f['org']} a-t-il démarré {f['product']} ?",
        f"Quelle est l'année de début de {f['product']} chez {f['org']} ?",
        f"En quelle année le projet {f['product']} chez {f['org']} a-t-il commencé ?",
    ])
    q_city = rng.choice([
        f"Dans quelle ville se trouve l'équipe {f['product']} chez {f['org']} ?",
        f"Dans quelle ville {f['org']} est-il situé pour le projet {f['product']} ?",
        f"Quelle ville est associée au projet {f['product']} de {f['org']} ?",
    ])
    return [
        {"qid": f"{doc_id}_q1", "question": q_year, "answer": str(f["year"])},
        {"qid": f"{doc_id}_q2", "question": q_city, "answer": f["city"]},
    ]


def zh_questions(doc_id: str, f: Dict, rng: random.Random) -> List[Dict]:
    q_year = rng.choice([
        f"{f['org']} 是在哪一年开始 {f['product']} 的？",
        f"{f['org']} 的 {f['product']} 项目启动年份是什么？",
        f"{f['product']} 在 {f['org']} 是哪一年启动的？",
    ])
    q_city = rng.choice([
        f"{f['org']} 的 {f['product']} 团队位于哪个城市？",
        f"{f['org']} 的 {f['product']} 项目在哪个城市？",
        f"{f['product']} 在 {f['org']} 对应的城市是哪里？",
    ])
    return [
        {"qid": f"{doc_id}_q1", "question": q_year, "answer": str(f["year"])},
        {"qid": f"{doc_id}_q2", "question": q_city, "answer": f["city"]},
    ]


# -----------------------
# Doc bundle generator
# -----------------------

def gen_doc_bundle(lang: str, doc_idx: int, rng: random.Random, group_key: Tuple[str, str, str]) -> Dict:
    """
    Hard negative grouping:
      group_key = (org, product, topic)
    Docs in same group share org/product/topic but differ in city/year/id/score/metric/value.
    """
    org, product, topic = group_key

    if lang == "en":
        city = rng.choice(EN_CITIES)
        metric_name = rng.choice(list(EN_METRICS.keys()))
        family = EN_METRICS[metric_name]
        value = metric_value(rng, family)
        templates = en_templates
        qgen = en_questions
    elif lang == "fr":
        city = rng.choice(FR_CITIES)
        metric_name = rng.choice(list(FR_METRICS.keys()))
        family = FR_METRICS[metric_name]
        value = metric_value(rng, family)
        templates = fr_templates
        qgen = fr_questions
    else:
        city = rng.choice(ZH_CITIES)
        metric_name = rng.choice(list(ZH_METRICS.keys()))
        family = ZH_METRICS[metric_name]
        value = metric_value(rng, family)
        templates = zh_templates
        qgen = zh_questions

    year = rand_year(rng)
    ident = rand_id(rng)
    score = rng.randint(15, 99)

    facts = {
        "org": org,
        "product": product,
        "topic": topic,
        "city": city,
        "year": year,
        "id": ident,
        "score": score,
        "metric": metric_name,
        "value": value,
    }

    doc_text = rng.choice(templates(facts))
    doc_id = f"{lang}_{doc_idx:04d}"

    return {
        "doc_id": doc_id,
        "language": lang,
        "document": doc_text,
        "questions": qgen(doc_id, facts, rng),
    }


def generate_dataset(total_docs: int = 200, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)

    base = total_docs // 3
    rem = total_docs % 3
    n_en = base + (1 if rem > 0 else 0)
    n_fr = base + (1 if rem > 1 else 0)
    n_zh = base

    def make_group_keys(lang: str, n_docs: int) -> List[Tuple[str, str, str]]:
        # create groups of 4 docs each
        if lang == "en":
            orgs, products, topics = EN_ORGS, EN_PRODUCTS, EN_TOPICS
        elif lang == "fr":
            orgs, products, topics = FR_ORGS, FR_PRODUCTS, FR_TOPICS
        else:
            orgs, products, topics = ZH_ORGS, ZH_PRODUCTS, ZH_TOPICS

        n_groups = max(1, n_docs // 4)
        keys = []
        for _ in range(n_groups):
            org = rng.choice(orgs)
            product = rng.choice(products)
            topic = rng.choice(topics)
            keys.append((org, product, topic))

        expanded = []
        while len(expanded) < n_docs:
            expanded.extend(keys)
        return expanded[:n_docs]

    dataset: List[Dict] = []

    for lang, n in [("en", n_en), ("fr", n_fr), ("zh", n_zh)]:
        group_keys = make_group_keys(lang, n)
        for i in range(1, n + 1):
            dataset.append(gen_doc_bundle(lang, i, rng, group_keys[i - 1]))

    rng.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    OUT_PATH = "data/qa_dataset_200.json"
    ensure_dir("data")

    dataset = generate_dataset(total_docs=200, seed=42)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    counts = {"en": 0, "fr": 0, "zh": 0}
    q_total = 0
    for item in dataset:
        counts[item["language"]] += 1
        q_total += len(item.get("questions", []))

    print(f"Wrote {len(dataset)} documents to {OUT_PATH}")
    print(f"Language counts: {counts}")
    print(f"Total questions: {q_total} (avg {q_total/len(dataset):.2f} per doc)")
    print("Example doc:\n", dataset[0]["document"])
