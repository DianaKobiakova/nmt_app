import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os


st.set_page_config(page_title="Калькулятор НМТ та Шанси на Вступ", layout="wide")

# --- КОНФІГУРАЦІЯ ПРЕДМЕТІВ ТА ШЛЯХІВ ДО МОДЕЛЕЙ ---
SUBJECTS_CONFIG = {
    "Українська мова": {
        "key": "new",
        "model_path": "src/lgbm_model_new.pkl"
    },
    "Математика": {
        "key": "math",
        "model_path": "src/lgbm_model_math.pkl"
    },
    "Історія України": {
        "key": "hist",
        "model_path": "src/lgbm_model_hist.pkl"
    }
}

# --- ОПЦІЇ ДЛЯ ВИПАДАЮЧИХ СПИСКІВ ---
settlement_types_options = ['обласний центр', 'місто', 'село', 'смт', 'інше']
school_types_options = [
    'середня загальноосвітня школа', 'навчально-виховний комплекс', 'ліцей',
    'спеціалізована школа', 'науковий ліцей', 'гімназія', 'заклад фахової передвищої освіти',
    'заклад вищої освіти', 'колегіум', 'заклад професійної (професійно-технічної) освіти',
    'загальноосвітня санаторна школа', "навчально-виховне об'єднання", 'ліцей із посиленою військово-фізичною підготовкою',
    'спортивний ліцей', 'середня загальноосвітня школа-інтернат', 'спеціалізована школа-інтернат',
    'спеціальна загальноосвітня школа', 'колегіум/колеж', 'військовий (військово-морський, військово-спортивний) ліцей',
    'колеж', 'вечірня (змінна) школа', 'спеціальна загальноосвітня школа-інтернат',
    'професійний ліцей відповідного профілю', 'початкова школа', 'Пенітенціарна установа',
    'мистецький ліцей', 'спеціальна школа', 'вищий навчальний заклад III-IV рівнів акредитації',
    'навчально-реабілітаційний центр', 'школа соціальної реабілітації', 'професійний коледж (коледж) спортивного профілю'
]
oblast_options = [
    'Миколаївська область', 'Черкаська область', 'Чернігівська область', 'Запорізька область', 'Луганська область',
    'Рівненська область', 'Одеська область', 'Київська область', 'Вінницька область', 'Тернопільська область',
    'Дніпропетровська область', 'м.Київ', 'Львівська область', 'Хмельницька область', 'Харківська область',
    'Кіровоградська область', 'Чернівецька область', 'Волинська область', 'Івано-Франківська область',
    'Донецька область', 'Полтавська область', 'Херсонська область', 'Закарпатська область', 'Сумська область',
    'Житомирська область'
]

# --- КОНСТАНТИ ДЛЯ РОЗРАХУНКІВ ---
NMT_MIN = 100.0
NMT_MAX = 200.0
S_MIN = 1.0
S_MAX = 12.0
DELTA_NMT = NMT_MAX - NMT_MIN
DELTA_S = S_MAX - S_MIN
O_AVG = 7.5
K_SCALE = DELTA_NMT / DELTA_S

# --- ЗАВАНТАЖЕННЯ МОДЕЛЕЙ НМТ---
@st.cache_resource
def load_all_nmt_models(subject_config):

    if 'dev' in os.environ['ENVIROMENT_MODE']:
        st.warning("Завантаження моделей НМТ вимкнено в режимі розробки. "
                   "Перевірте, чи встановлено змінну оточення ENVIROMENT_MODE у 'prod' для завантаження моделей.")
    elif 'prod' in os.environ['ENVIROMENT_MODE']:
        st.info("Завантаження моделей НМТ увімкнено 'prod'.")
        import boto3
        s3 = boto3.client('s3')
        s3.download_file('nmt', 'lgbm_model_hist.pkl', 'lgbm_model_hist.pkl')
        s3.download_file('nmt', 'lgbm_model_math.pkl', 'lgbm_model_math.pkl')
        s3.download_file('nmt', 'lgbm_model_new.pkl', 'lgbm_model_new.pkl')
        s3.download_file('nmt', 'konkurs_NMT.csv', 'konkurs_NMT.csv')



    loaded_models = {}
    all_loaded_successfully = True
    for subject_display_name, config in subject_config.items():
        subject_key = config["key"]
        model_path = config["model_path"]
        try:
            loaded_models[subject_key] = joblib.load(model_path)
        except FileNotFoundError:
            st.error(f"ПОМИЛКА: Файл моделі {model_path} для '{subject_display_name}' НЕ ЗНАЙДЕНО.")
            loaded_models[subject_key] = None
            all_loaded_successfully = False
        except Exception as e:
            st.error(f"ПОМИЛКА завантаження моделі {model_path} для '{subject_display_name}': {e}")
            loaded_models[subject_key] = None
            all_loaded_successfully = False
    return loaded_models, all_loaded_successfully

nmt_models, all_nmt_models_loaded = load_all_nmt_models(SUBJECTS_CONFIG)
if all_nmt_models_loaded and nmt_models:
    st.sidebar.success("Моделі НМТ завантажено!")
else:
    st.sidebar.error("Помилка завантаження моделей НМТ!")


# --- ФУНКЦІЇ РОЗРАХУНКУ БАЛІВ НМТ ---
def calculate_score_balanced(b_model: float, o_12: float, w: float = 0.5) -> float:
    if not (S_MIN <= o_12 <= S_MAX): return max(NMT_MIN, min(NMT_MAX, b_model))
    b_o12_norm = NMT_MIN + (o_12 - S_MIN) * (DELTA_NMT / DELTA_S)
    final_score = w * b_model + (1 - w) * b_o12_norm
    return max(NMT_MIN, min(NMT_MAX, final_score))

def calculate_score_individual_adjusted(b_model: float, o_12: float) -> float:
    if not (S_MIN <= o_12 <= S_MAX): return max(NMT_MIN, min(NMT_MAX, b_model))
    b_adjusted = b_model + (o_12 - O_AVG) * K_SCALE
    final_score = max(NMT_MIN, min(NMT_MAX, b_adjusted))
    return final_score

def calculate_score_cautious_stress(b_model: float, o_12: float, k_stress: float = 1.0) -> float:
    if not (S_MIN <= o_12 <= S_MAX): return max(NMT_MIN, min(NMT_MAX, b_model))
    o_12_stressed = max(S_MIN, o_12 - k_stress)
    b_o12_stressed_norm = NMT_MIN + (o_12_stressed - S_MIN) * (DELTA_NMT / DELTA_S)
    final_score = (b_model + b_o12_stressed_norm) / 2
    return max(NMT_MIN, min(NMT_MAX, final_score))

# --- ФУНКЦІЇ ДЛЯ АНАЛІЗУ ШАНСІВ НА ВСТУП ---
@st.cache_data
def load_university_data(data_path):
    try:
        df = pd.read_csv(data_path)
        
        col_university_orig = 'Назва закладу'
        col_specialty_orig = 'Спеціальність'
        col_min_score_orig = 'шк_Мін. бал\n(на загальних підставах)'
        col_avg_score_orig = 'шк_Сер. бал\n(на загальних підставах)'
        col_max_score_orig = 'шк_Макс. бал\n(на загальних підставах)'
        col_degree_level_orig = 'Освітній ступінь'
        col_basis_of_entry_orig = 'Вступ на основі'
        col_form_of_study_orig = 'Форма навчання'

        required_columns_for_chances = {
            col_university_orig, col_specialty_orig, 
            col_min_score_orig, col_avg_score_orig, col_max_score_orig
        }
        
        missing_cols = required_columns_for_chances - set(df.columns)
        if missing_cols:
            st.error(f"CSV файл '{data_path}' не містить обов'язкових колонок: {', '.join(missing_cols)}")
            return None

        df_processed = df.copy()
        score_cols_to_convert = [col_min_score_orig, col_avg_score_orig, col_max_score_orig]
        for col in score_cols_to_convert:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].str.replace(',', '.', regex=False)
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        df_processed.dropna(subset=score_cols_to_convert, inplace=True)
        if df_processed.empty:
            st.warning(f"Файл '{data_path}' порожній після видалення рядків з некоректними балами.")
            return None
        
        rename_map = {
            col_university_orig: 'Університет', col_specialty_orig: 'Спеціальність',
            col_min_score_orig: 'Мін_Бал', col_avg_score_orig: 'Сер_Бал', col_max_score_orig: 'Макс_Бал'
        }
        optional_cols_map = {
            col_degree_level_orig: 'Освітній_ступінь', col_basis_of_entry_orig: 'Вступ_на_основі',
            col_form_of_study_orig: 'Форма_навчання'
        }
        for orig_col, new_col in optional_cols_map.items():
            if orig_col in df_processed.columns: rename_map[orig_col] = new_col
        
        df_processed.rename(columns=rename_map, inplace=True)

        grouping_keys = ['Університет', 'Спеціальність']
        for col_name in ['Освітній_ступінь', 'Вступ_на_основі', 'Форма_навчання']:
            if col_name in df_processed.columns:
                grouping_keys.append(col_name)
                df_processed[col_name].fillna('Не вказано', inplace=True) 

        agg_funcs = {'Мін_Бал': 'mean', 'Сер_Бал': 'mean', 'Макс_Бал': 'mean'}
        df_aggregated = df_processed.groupby(grouping_keys, as_index=False).agg(agg_funcs)
        
        for col in ['Мін_Бал', 'Сер_Бал', 'Макс_Бал']:
            df_aggregated[col] = df_aggregated[col].round(2)

        final_columns_to_keep = [col for col in grouping_keys + ['Мін_Бал', 'Сер_Бал', 'Макс_Бал'] if col in df_aggregated.columns]
        return df_aggregated[final_columns_to_keep]

    except FileNotFoundError:
        st.error(f"Файл '{data_path}' не знайдено. Перевірте шлях та наявність файлу.")
        return None
    except pd.errors.EmptyDataError:
        st.error(f"Файл '{data_path}' порожній.")
        return None
    except KeyError as e:
        st.error(f"Помилка ключа при обробці даних: колонка {e} не знайдена. Перевірте відповідність назв колонок у файлі та в коді.")
        return None
    except Exception as e:
        st.error(f"Помилка при завантаженні або обробці файлу '{data_path}': {e}")
        return None

def get_admission_chances(applicant_score, min_score, avg_score, max_score):
    if applicant_score is None: return "Н/Д (немає балу абітурієнта)"
    if pd.isna(min_score) or pd.isna(avg_score) or pd.isna(max_score): return "Н/Д (немає даних по спеціальності)"
    if applicant_score >= max_score: return "🏆 Дуже високий шанс (вище макс.)"
    elif applicant_score >= avg_score + (max_score - avg_score) * 0.75 : return "🥇 Дуже високий шанс"
    elif applicant_score >= avg_score + (max_score - avg_score) * 0.25: return "🥈 Високий шанс"
    elif applicant_score >= avg_score: return "🥉 Хороший шанс"
    elif applicant_score >= min_score + (avg_score - min_score) * 0.75: return "👍 Задовільний шанс"
    elif applicant_score >= min_score: return "😐 Середній шанс (конкурсна)"
    elif applicant_score >= min_score * 0.95 : return "⚠️ Низький шанс (на межі)"
    elif applicant_score >= min_score * 0.9: return "📉 Дуже низький шанс"
    else: return "📉📉 Вкрай низький шанс"

# Порядок шансів для сортування та фільтрації
CHANCE_ORDER_MAP = {
    "🏆 Дуже високий шанс (вище макс.)": 0, "🥇 Дуже високий шанс": 1, "🥈 Високий шанс": 2,
    "🥉 Хороший шанс": 3, "👍 Задовільний шанс": 4, "😐 Середній шанс (конкурсна)": 5,
    "⚠️ Низький шанс (на межі)": 6, "📉 Дуже низький шанс": 7, "📉📉 Вкрай низький шанс": 8,
    "Н/Д (немає даних по спеціальності)": 9, "Н/Д (немає балу абітурієнта)": 10
}


# --- ОСНОВНИЙ ІНТЕРФЕЙС З ВКЛАДКАМИ ---
st.title("🧮 Калькулятор НМТ та Аналіз Шансів на Вступ 🎓")

if 'applicant_total_score' not in st.session_state: st.session_state.applicant_total_score = None
if 'calculated_subject_scores_display' not in st.session_state: st.session_state.calculated_subject_scores_display = {}

tab1, tab2 = st.tabs(["📊 Розрахунок балу НМТ", "🎓 Аналіз шансів на вступ"])

with tab1:
    st.markdown("""
    ### Ласкаво просимо до Калькулятора приблизного балу НМТ!
    Цей інструмент допоможе вам отримати **орієнтовну оцінку** можливих результатів Національного Мультипредметного Тесту (НМТ)
    на основі загальних даних та ваших шкільних оцінок.

    #### Важливо пам'ятати:
    * Розрахунки є **статистично приблизними** і не гарантують точного результату.
    * Реальний бал НМТ залежить від вашої індивідуальної підготовки, рівня складності завдань у день тестування та інших факторів.
    * Цей калькулятор **не є офіційним інструментом** Українського центру оцінювання якості освіти (УЦОЯО).
    """)
    st.markdown("---")

    if not all_nmt_models_loaded:
        st.error("Не вдалося завантажити одну або більше моделей для розрахунку НМТ. Розрахунок балів НМТ не доступний.")
        st.stop()

    st.header("🙋 Загальна інформація про абітурієнта")
    col1, col2 = st.columns(2)
    with col1:
        exam_year = st.number_input("Рік складання НМТ", min_value=2022, max_value=2070, value=st.session_state.get('exam_year_val', 2025), step=1, help="Рік, у якому планується або відбулося складання НМТ.")
        sextypename_options = ['чоловіча', 'жіноча']
        current_sextypename = st.session_state.get('sextypename_val', sextypename_options[0])
        sextypename = st.radio("Стать", options=sextypename_options, horizontal=True, index=sextypename_options.index(current_sextypename), help="Ваша стать.")
        current_regname = st.session_state.get('regname_val', oblast_options[0])
        regname = st.selectbox('Область реєстрації', options=oblast_options, index=oblast_options.index(current_regname) if current_regname in oblast_options else 0, help="Область, де ви зареєстровані або де знаходиться ваш навчальний заклад.")
    with col2:
        birth = st.number_input("Рік народження", min_value=1950, max_value=2020, value=st.session_state.get('birth_val', 2008), step=1, help="Ваш повний рік народження.")
        current_settlement_type = st.session_state.get('settlement_type_val', settlement_types_options[0])
        settlement_type = st.selectbox('Тип населеного пункту', options=settlement_types_options, index=settlement_types_options.index(current_settlement_type) if current_settlement_type in settlement_types_options else 0, help="Тип населеного пункту вашого навчального закладу.")
        current_eotypename = st.session_state.get('eotypename_val', school_types_options[0])
        eotypename = st.selectbox('Тип закладу освіти', options=school_types_options, index=school_types_options.index(current_eotypename) if current_eotypename in school_types_options else 0, help="Тип вашого закладу освіти.")
    st.markdown("---")

    st.header("📚 Шкільні оцінки за предмети")
    st.caption("Введіть ваші річні (або атестаційні) оцінки за 12-бальною шкалою.")
    o12_scores_input = {}
    cols_subjects = st.columns(len(SUBJECTS_CONFIG))
    for idx, (subject_display_name, config) in enumerate(SUBJECTS_CONFIG.items()):
        with cols_subjects[idx]:
            subject_key = config["key"]
            o12_scores_input[subject_key] = st.number_input(
                f"Оцінка '{subject_display_name}' (1-12):",
                min_value=S_MIN, max_value=S_MAX, value=st.session_state.get(f"o12_{subject_key}_val", 8.0), step=0.5,
                key=f"o12_{subject_key}", help=f"Ваша шкільна оцінка з предмету '{subject_display_name}'."
            )
    st.markdown("---")

    with st.expander("⚙️ Додаткові налаштування та деталі розрахунку НМТ", expanded=True):
        st.markdown("""
        Ці параметри дозволяють вам варіювати розрахунок деяких прогнозних балів, щоб побачити різні сценарії.
        Наведені нижче формули є прикладами того, як можуть враховуватися різні фактори.
        """)
        w_formula1 = st.slider("Вага прогнозу моделі (для 'Баланс: Прогноз та Шкільна успішність'):", 0.0, 1.0, st.session_state.get('w_formula1_val', 0.5), 0.05, key="w_slider",
                               help="Визначає вплив прогнозу моделі порівняно зі шкільною оцінкою (0.0 - тільки шкільна оцінка, 1.0 - тільки прогноз моделі).")
        k_stress_formula3 = st.slider("Фактор стресу (для 'Обережний прогноз'):", 0.0, 3.0, st.session_state.get('k_stress_formula3_val', 1.0), 0.1, key="k_stress_slider",
                                  help="Імітує вплив стресу, знижуючи шкільну оцінку на вказану кількість балів перед розрахунком.")
        st.markdown("---")
        st.subheader("Деталізація формул (для довідки):")
        st.markdown(r"""
        **1. Баланс: Прогноз та Шкільна успішність ($B_{final,1}$)**
        Враховує як прогноз моделі ($B_{model}$), так і вашу шкільну успішність ($O_{12}$), нормалізовану до шкали НМТ.
        $B_{O12\_norm} = NMT_{MIN} + (O_{12} - S_{MIN}) \times \frac{NMT_{MAX} - NMT_{MIN}}{S_{MAX} - S_{MIN}}$
        $B_{final,1} = w \times B_{model} + (1-w) \times B_{O12\_norm}$
        де $w$ - вага прогнозу моделі.

        **2. Індивідуальний прогноз (з поправкою на шкільні оцінки, $B_{final,2}$)**
        Базується на прогнозі моделі ($B_{model}$), скоригованому на різницю між вашою шкільною оцінкою ($O_{12}$) та середньостатистичною ($O_{AVG}$), масштабованою до діапазону НМТ.
        $K_{scale} = \frac{NMT_{MAX} - NMT_{MIN}}{S_{MAX} - S_{MIN}}$
        $B_{adjusted} = B_{model} + (O_{12} - O_{AVG}) \times K_{scale}$
        $B_{final,2} = \max(NMT_{MIN}, \min(NMT_{MAX}, B_{adjusted}))$

        **3. Обережний прогноз (з урахуванням фактору стресу, $B_{final,3}$)**
        Дає більш обережну оцінку, враховуючи можливий вплив екзаменаційного стресу ($k_{stress}$), який "знижує" шкільну оцінку.
        $O_{12\_stressed} = \max(S_{MIN}, O_{12} - k_{stress})$
        $B_{O12\_stressed\_norm} = NMT_{MIN} + (O_{12\_stressed} - S_{MIN}) \times \frac{NMT_{MAX} - NMT_{MIN}}{S_{MAX} - S_{MIN}}$
        $B_{final,3} = \frac{B_{model} + B_{O12\_stressed\_norm}}{2}$
        """)
    st.markdown("---")

    if st.button("📈 Розрахувати приблизні бали НМТ", type="primary", use_container_width=True):
        st.session_state.exam_year_val = exam_year
        st.session_state.sextypename_val = sextypename
        st.session_state.regname_val = regname
        st.session_state.birth_val = birth
        st.session_state.settlement_type_val = settlement_type
        st.session_state.eotypename_val = eotypename
        st.session_state.w_formula1_val = w_formula1
        st.session_state.k_stress_formula3_val = k_stress_formula3
        for sk_key, val_o12 in o12_scores_input.items():
             st.session_state[f"o12_{sk_key}_val"] = val_o12

        all_grades_valid = True
        for subject_name_iter, config_iter in SUBJECTS_CONFIG.items():
            subject_key_iter_val = config_iter["key"]
            current_grade = o12_scores_input[subject_key_iter_val]
            if not (S_MIN <= current_grade <= S_MAX):
                st.error(f"Некоректна шкільна оцінка '{current_grade}' для '{subject_name_iter}'. Діапазон: [{S_MIN:.1f}-{S_MAX:.1f}].")
                all_grades_valid = False
        
        if not all_grades_valid:
            st.session_state.applicant_total_score = None
            st.session_state.calculated_subject_scores_display = {}
            st.stop()

        try:
            feature_cols = ['exam_year', 'birth', 'sextypename', 'regname', 'settlement_type', 'eotypename']
            input_values = [exam_year, birth, sextypename, regname, settlement_type, eotypename]
            common_input_data = pd.DataFrame([input_values], columns=feature_cols)

            st.header("📊 Результати розрахунку по предметах:")
            average_subject_scores_for_total = []
            st.session_state.calculated_subject_scores_display = {}
            subject_cols = st.columns(len(SUBJECTS_CONFIG))

            calculation_successful_for_at_least_one = False
            for idx, (subject_display_name, config_item) in enumerate(SUBJECTS_CONFIG.items()):
                with subject_cols[idx]:
                    subject_key = config_item["key"]
                    model_subject = nmt_models.get(subject_key)
                    o_12_subject = o12_scores_input[subject_key]

                    if model_subject is None:
                        st.warning(f"Модель для '{subject_display_name}' не завантажена.")
                        st.session_state.calculated_subject_scores_display[subject_display_name] = "Модель не завантажена"
                        continue

                    st.subheader(f"{subject_display_name}")
                    predicted_score_subject = model_subject.predict(common_input_data)[0]
                    score_1 = calculate_score_balanced(predicted_score_subject, o_12_subject, w_formula1)
                    score_2 = calculate_score_individual_adjusted(predicted_score_subject, o_12_subject)
                    score_3 = calculate_score_cautious_stress(predicted_score_subject, o_12_subject, k_stress_formula3)
                    avg_subj_score = (score_1 + score_2 + score_3) / 3
                    average_subject_scores_for_total.append(avg_subj_score)
                    calculation_successful_for_at_least_one = True

                    st.metric(label="1. Баланс", value=f"{score_1:.2f}")
                    st.metric(label="2. Індивідуальний", value=f"{score_2:.2f}")
                    st.metric(label="3. Обережний", value=f"{score_3:.2f}")
                    st.markdown("---")
                    st.metric(label=f"Середній з предмету:", value=f"{avg_subj_score:.2f}", delta_color="off")
                    st.session_state.calculated_subject_scores_display[subject_display_name] = {
                        "Прогноз моделі": f"{predicted_score_subject:.2f}", "Баланс": f"{score_1:.2f}",
                        "Індивідуальний": f"{score_2:.2f}", "Обережний": f"{score_3:.2f}",
                        "Середній з предмету": f"{avg_subj_score:.2f}"
                    }
            st.markdown("---")

            if calculation_successful_for_at_least_one and average_subject_scores_for_total:
                final_total_nmt_score = sum(average_subject_scores_for_total) / len(average_subject_scores_for_total)
                st.session_state.applicant_total_score = final_total_nmt_score
                
                st.subheader("🏆 Ваш УЗАГАЛЬНЕНИЙ СЕРЕДНІЙ бал НМТ:")
                st.metric(label="Середній бал НМТ (100-200)", value=f"{final_total_nmt_score:.2f}",
                              help="Це середнє арифметичне від середніх балів кожного предмету. Шкала 100-200.")
                st.toast('🚀 Розрахунок успішно завершено!', icon='🎉')
            else:
                st.error("Не вдалося розрахувати бали для жодного предмету (можливо, моделі не завантажені або дані некоректні).")
                st.session_state.applicant_total_score = None
        except Exception as e:
            st.error(f"Сталася непередбачена помилка під час розрахунку НМТ: {e}")
            st.session_state.applicant_total_score = None
            st.session_state.calculated_subject_scores_display = {}
            st.warning("Будь ласка, перевірте введені дані.")
            
    elif st.session_state.applicant_total_score is not None and st.session_state.calculated_subject_scores_display:
        st.success("Бали НМТ вже були розраховані. Результати нижче:")
        # ... (код відображення результатів з session_state) ...
        st.header("📊 Попередні результати розрахунку по предметах:")
        subject_cols_results = st.columns(len(SUBJECTS_CONFIG))
        for idx, (subject_display_name, config_item) in enumerate(SUBJECTS_CONFIG.items()):
            with subject_cols_results[idx]:
                st.subheader(f"{subject_display_name}")
                data = st.session_state.calculated_subject_scores_display.get(subject_display_name)
                if isinstance(data, dict):
                    st.metric(label="1. Баланс", value=data.get("Баланс", "Н/Д"))
                    st.metric(label="2. Індивідуальний", value=data.get("Індивідуальний", "Н/Д"))
                    st.metric(label="3. Обережний", value=data.get("Обережний", "Н/Д"))
                    st.markdown("---")
                    st.metric(label=f"Середній з предмету:", value=data.get("Середній з предмету", "Н/Д"))
                else:
                    st.warning(str(data))
        st.markdown("---")
        st.subheader("🏆 Ваш УЗАГАЛЬНЕНИЙ СЕРЕДНІЙ бал НМТ:")
        st.metric(label="Середній бал НМТ (100-200)", value=f"{st.session_state.applicant_total_score:.2f}")

with tab2:
    st.header("Аналіз шансів на вступ до університетів")
    default_file_name = "src/konkurs_NMT.csv"

    if st.session_state.applicant_total_score is None:
        st.warning("⚠️ Будь ласка, спочатку розрахуйте ваш узагальнений середній бал НМТ на вкладці 'Розрахунок балу НМТ'.")
        st.info(f"Для аналізу буде використано дані з файлу: `{default_file_name}` (якщо він доступний).")
    else:
        st.success(f"Ваш узагальнений СЕРЕДНІЙ бал НМТ для аналізу: **{st.session_state.applicant_total_score:.2f}** (шкала 100-200)")
        st.markdown("---")
        st.caption("Цей розділ дозволяє вам проаналізувати ваші шанси на вступ до різних університетів на основі розрахованого балу НМТ.")
        st.caption("Використовуйте фільтри для вибору університетів, спеціальностей та інших параметрів.")
        st.markdown("---")

        university_df = load_university_data(default_file_name)

        if university_df is not None and not university_df.empty:
            st.subheader("Фільтри та результати аналізу:")
            st.caption(f"Дані для аналізу завантажено з файлу: `{default_file_name}`. Усереднено за роками для однакових конкурсних пропозицій.")
            
            # Основні фільтри
            filter_cols_1, filter_cols_2, filter_cols_3 = st.columns(3)
            with filter_cols_1:
                unique_universities = sorted(university_df['Університет'].unique())
                selected_universities = st.multiselect("Університет(и):", unique_universities, placeholder="Всі університети", key="uni_filter")
            
            # Розраховуємо initial_results_df після первинних фільтрів даних
            active_filters_df = university_df.copy()
            if selected_universities:
                 active_filters_df = active_filters_df[active_filters_df['Університет'].isin(selected_universities)]
            
            with filter_cols_2:
                if 'Освітній_ступінь' in active_filters_df.columns:
                    unique_degree_levels = sorted(active_filters_df['Освітній_ступінь'].dropna().unique())
                    selected_degree_levels = st.multiselect("Освітній ступінь:", unique_degree_levels, placeholder="Всі ступені", key="deg_filter")
                    if selected_degree_levels: active_filters_df = active_filters_df[active_filters_df['Освітній_ступінь'].isin(selected_degree_levels)]

                if 'Форма_навчання' in active_filters_df.columns:
                    unique_study_forms = sorted(active_filters_df['Форма_навчання'].dropna().unique())
                    selected_study_forms = st.multiselect("Форма навчання:", unique_study_forms, placeholder="Всі форми", key="form_filter")
                    if selected_study_forms: active_filters_df = active_filters_df[active_filters_df['Форма_навчання'].isin(selected_study_forms)]
            
            with filter_cols_3:
                if 'Вступ_на_основі' in active_filters_df.columns:
                    unique_basis_of_entry = sorted(active_filters_df['Вступ_на_основі'].dropna().unique())
                    default_basis = "Повна загальна середня освіта" if "Повна загальна середня освіта" in unique_basis_of_entry else None
                    default_basis_list = [default_basis] if default_basis else []
                    selected_basis_of_entry = st.multiselect("Вступ на основі:", unique_basis_of_entry, default=default_basis_list, placeholder="Всі основи", key="basis_filter")
                    if selected_basis_of_entry: active_filters_df = active_filters_df[active_filters_df['Вступ_на_основі'].isin(selected_basis_of_entry)]

                unique_specialties = sorted(active_filters_df['Спеціальність'].dropna().unique())
                selected_specialties = st.multiselect("Спеціальність(і):", unique_specialties, placeholder="Всі спеціальності", key="spec_filter")
                if selected_specialties: active_filters_df = active_filters_df[active_filters_df['Спеціальність'].isin(selected_specialties)]

            # Розрахунок шансів для попередньо відфільтрованих даних
            results_df_for_chances = active_filters_df.copy()
            if not results_df_for_chances.empty:
                results_df_for_chances['Шанс Вступу'] = results_df_for_chances.apply(
                    lambda row: get_admission_chances(st.session_state.applicant_total_score, row['Мін_Бал'], row['Сер_Бал'], row['Макс_Бал']),
                    axis=1
                )

                # Фільтр за розрахованим шансом вступу
                # Цей фільтр має бути після розрахунку 'Шанс Вступу'
                st.markdown("---") # Розділювач перед фільтром шансів
                
                # Сортуємо рівні шансів для коректного відображення у фільтрі
                # Використовуємо CHANCE_ORDER_MAP для отримання правильного порядку
                unique_chance_levels_calculated = sorted(
                    results_df_for_chances['Шанс Вступу'].dropna().unique(),
                    key=lambda x: CHANCE_ORDER_MAP.get(x, 99) # 99 для невідомих значень, щоб вони були в кінці
                )
                selected_chance_levels = st.multiselect(
                    "Фільтр за рівнем шансів:", 
                    options=unique_chance_levels_calculated, 
                    placeholder="Показати всі рівні шансів",
                    key="chance_level_filter"
                )

                final_results_df = results_df_for_chances
                if selected_chance_levels:
                    final_results_df = final_results_df[final_results_df['Шанс Вступу'].isin(selected_chance_levels)]
                
                if not final_results_df.empty:
                    final_results_df['Сортування_Шансів'] = final_results_df['Шанс Вступу'].map(CHANCE_ORDER_MAP)
                    results_df_sorted = final_results_df.sort_values(by=['Сортування_Шансів', 'Університет', 'Спеціальність']).drop(columns=['Сортування_Шансів'])
                    
                    display_columns = ['Університет', 'Спеціальність', 'Мін_Бал', 'Сер_Бал', 'Макс_Бал', 'Шанс Вступу']
                    insert_pos = 2 
                    if 'Освітній_ступінь' in results_df_sorted.columns: 
                        display_columns.insert(insert_pos, 'Освітній_ступінь'); insert_pos+=1
                    if 'Форма_навчання' in results_df_sorted.columns: 
                        display_columns.insert(insert_pos, 'Форма_навчання'); insert_pos+=1
                    if 'Вступ_на_основі' in results_df_sorted.columns: 
                        display_columns.insert(insert_pos, 'Вступ_на_основі'); insert_pos+=1
                    
                    # Зміна для компактності таблиці
                    st.dataframe(results_df_sorted[display_columns], height=500, use_container_width=False, hide_index=True)
                    st.markdown("---")
                    st.info(
                        """
                        **Як інтерпретувати результати шансів:**
                        Порівняння вашого **середнього балу НМТ (100-200)** з **середніми балами для вступу (100-200)** минулих років.
                        "Н/Д" означає відсутність даних. *Це **приблизна оцінка**.*
                        """
                    )
                else:
                    st.info("Не знайдено пропозицій за обраними фільтрами (включаючи фільтр за рівнем шансів).")
            else:
                st.info("Не знайдено пропозицій за обраними первинними фільтрами (університет, спеціальність тощо).")
        elif university_df is None :
             st.error(f"Не вдалося завантажити або обробити файл даних університетів: '{default_file_name}'. Перевірте шлях, наявність та коректність файлу.")

st.sidebar.header("ℹ️ Про проєкт")
st.sidebar.info(
    """
    Цей інструмент допоможе абітурієнтам:
    1. Розрахувати приблизний **середній бал НМТ (100-200)** з трьох предметів.
    2. Оцінити свої шанси на вступ до різних ЗВО України.
    **Важливо:** Розрахунки є **статистично приблизними** і не гарантують точного результату.
    Дані в файлі університетів усереднюються за роками для однакових конкурсних пропозицій.
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("Бажаємо успіху на НМТ та при вступі!")
st.sidebar.caption("Зроблено з ❤️ для українських абітурієнтів!")