import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# import datetime # Not explicitly used in the provided snippet, but can be kept if needed elsewhere

st.set_page_config(layout="wide", page_title="Дашборди аналізу даних тестування")


@st.cache_data  # Декоратор Streamlit для кешування даних
def load_and_preprocess_data(file_path):
    """
    Завантажує дані з CSV файлу, виконує початкову обробку (наприклад, конвертацію дат)
    та кешує результат.
    """
    try:
        df = pd.read_csv(file_path)
        # Початкова обробка даних, яка виконується лише один раз
        if 'testdate' in df.columns:
            df['testdate'] = pd.to_datetime(df['testdate'], errors='coerce')
        # Сюди можна додати іншу статичну обробку, якщо вона потрібна для main_df
        return df
    except FileNotFoundError:
        st.error(f"Помилка: Файл '{file_path}' не знайдено. Перевірте шлях до файлу.")
        st.stop()  # Зупиняємо виконання, якщо файл не знайдено
    except Exception as e:
        st.error(f"Помилка при завантаженні або початковій обробці даних з '{file_path}': {e}")
        st.stop() # Зупиняємо виконання при інших помилках завантаження/обробки

# --- Завантаження основних даних з використанням кешованої функції ---
# Тепер замість прямого pd.read_csv, ми викликаємо нашу кешовану функцію:
main_df = load_and_preprocess_data('src/main_df.csv')

# --- Додаткова перевірка після завантаження (опціонально, але корисно) ---
# Функція load_and_preprocess_data вже використовує st.stop() у випадку критичних помилок.
# Ця перевірка може бути корисною, якщо CSV файл порожній, але коректно завантажився.
if main_df.empty:
    st.warning("Увага: Файл даних ('src/main_df.csv') завантажено, але він порожній (не містить записів). Деякі елементи дашборду можуть не відображатися або відображатися некоректно.")
    # Розгляньте, чи потрібно тут st.stop(), чи дашборд може продовжувати роботу,
    # показуючи повідомлення "немає даних" у відповідних місцях.


st.title("🚀 Аналітичні дашборди на основі даних тестування")
st.markdown("Огляд даних минулих років. Використовуйте фільтри на бічній панелі для деталізації.")


# --- Sidebar Filters ---
st.sidebar.header("⚙️ Глобальні фільтри")

# Ensure main_df columns are available before using them for filters
if 'exam_year' not in main_df.columns or 'regname' not in main_df.columns:
    st.sidebar.error("Критична помилка: Необхідні колонки ('exam_year', 'regname') відсутні в даних.")
    st.stop()

selected_exam_year = st.sidebar.multiselect(
    "Виберіть рік іспиту:",
    options=sorted(main_df['exam_year'].unique(), reverse=True),
    default=list(sorted(main_df['exam_year'].unique())) # Ensure default is a list
)

unique_regions = sorted(main_df['regname'].unique())
selected_region = st.sidebar.multiselect(
    "Виберіть регіон:",
    options=unique_regions,
    default=[] 
)

# --- Filter Data ---
filtered_df = main_df.copy()
if selected_exam_year:
    filtered_df = filtered_df[filtered_df['exam_year'].isin(selected_exam_year)]
if selected_region: 
    filtered_df = filtered_df[filtered_df['regname'].isin(selected_region)]

# --- CENTRALIZED 'Вік' (AGE) CALCULATION ---
age_calculation_possible = True
if not filtered_df.empty:
    if 'birth' in filtered_df.columns and 'exam_year' in filtered_df.columns:
        birth_years = pd.to_numeric(filtered_df['birth'], errors='coerce')
        exam_years_numeric = pd.to_numeric(filtered_df['exam_year'], errors='coerce')
        # Add 'Вік' column to filtered_df
        filtered_df['Вік'] = exam_years_numeric - birth_years
    else:
        st.sidebar.warning("Колонки 'birth' або 'exam_year' відсутні. Віковий розподіл не буде розраховано.")
        # Add an empty 'Вік' column with NaN to prevent KeyErrors if plots expect it
        filtered_df['Вік'] = np.nan 
        age_calculation_possible = False
else:
    # If filtered_df is empty, still ensure 'Вік' column exists if other code expects it
    if 'Вік' not in filtered_df.columns:
         filtered_df['Вік'] = pd.Series(dtype='float64')


# --- Main Page Content ---
if filtered_df.empty and not (selected_exam_year or selected_region): # Check if empty due to no data initially
    st.warning("😔 Вхідний файл даних порожній або не містить записів.")
elif filtered_df.empty: # Empty due to filters
    st.warning("😔 Немає даних для вибраних фільтрів. Спробуйте змінити параметри.")
else:
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧑‍🤝‍🧑 Демографія",
        "📈 Тенденції тестування",
        "🗺️ Географія",
        "🏫 Заклади та пункти"
    ])

    # --- 1. Демографічний огляд ---
    with tab1:
        st.header("🧑‍🤝‍🧑 Демографічний огляд учасників")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Розподіл за статтю (`sextypename`)")
            if 'sextypename' in filtered_df.columns and not filtered_df['sextypename'].dropna().empty:
                gender_counts = filtered_df['sextypename'].value_counts().reset_index()
                gender_counts.columns = ['Стать', 'Кількість']
                fig_gender = px.pie(gender_counts, values='Кількість', names='Стать', title="Співвідношення за статтю", hole=0.3)
                fig_gender.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_gender, use_container_width=True)
            else:
                st.info("Немає даних для розподілу за статтю.")

            st.subheader("Розподіл за типом населеного пункту (`settlement_type`)")
            if 'settlement_type' in filtered_df.columns and not filtered_df['settlement_type'].dropna().empty:
                settlement_counts = filtered_df['settlement_type'].value_counts().reset_index()
                settlement_counts.columns = ['Тип населеного пункту', 'Кількість']
                fig_settlement = px.bar(settlement_counts, x='Тип населеного пункту', y='Кількість',
                                        title="Учасники за типом населеного пункту", color='Тип населеного пункту',
                                        labels={'Кількість':'Кількість учасників'})
                st.plotly_chart(fig_settlement, use_container_width=True)
            else:
                st.info("Немає даних для розподілу за типом населеного пункту.")

        with col2: # Ensure this is the ONLY 'with col2:' block in tab1
            st.subheader("Віковий розподіл (на основі `birth` та `exam_year`)")
            if age_calculation_possible and 'Вік' in filtered_df.columns and not filtered_df['Вік'].isnull().all():
                # Use a copy for plotting to handle NaNs and type conversion locally
                age_plot_data = filtered_df.dropna(subset=['Вік']).copy()
                if not age_plot_data.empty:
                    age_plot_data['Вік'] = age_plot_data['Вік'].astype(int)
                    fig_age = px.histogram(age_plot_data, x='Вік', nbins=30, title="Розподіл учасників за віком", marginal="box")
                    st.plotly_chart(fig_age, use_container_width=True)
                else:
                    st.info("Немає дійсних даних для вікового розподілу після обробки.")
            elif not age_calculation_possible:
                 st.info("Віковий розподіл не може бути показаний, оскільки колонки 'birth' або 'exam_year' відсутні.")
            else:
                st.info("Немає даних для вікового розподілу.")

            st.subheader("Розподіл за регіоном (`regname`)")
            if 'regname' in filtered_df.columns and not filtered_df['regname'].dropna().empty:
                region_counts = filtered_df['regname'].value_counts().reset_index()
                region_counts.columns = ['Регіон', 'Кількість']
                fig_region = px.bar(region_counts.sort_values('Кількість', ascending=False),
                                    x='Регіон', y='Кількість', title="Кількість учасників по регіонах", color='Регіон')
                st.plotly_chart(fig_region, use_container_width=True)
            else:
                st.info("Немає даних для розподілу за регіоном.")
    
    # --- 2. Аналіз тенденцій тестування ---
    with tab2:
        st.header("📈 Аналіз тенденцій тестування")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Кількість тестувань за роками (`exam_year`)")
            if 'exam_year' in main_df.columns and not main_df['exam_year'].dropna().empty:
                yearly_tests_all_data = main_df['exam_year'].value_counts().sort_index().reset_index()
                yearly_tests_all_data.columns = ['Рік', 'Кількість']
                fig_yearly_tests = px.line(yearly_tests_all_data, x='Рік', y='Кількість', markers=True, title="Динаміка кількості тестувань по роках")
                fig_yearly_tests.update_xaxes(type='category') 
                st.plotly_chart(fig_yearly_tests, use_container_width=True)
            else:
                st.info("Немає даних 'exam_year' для відображення динаміки по роках.")

            st.subheader("Розподіл за типом реєстрації (`regtypename`)")
            if 'regtypename' in filtered_df.columns and not filtered_df['regtypename'].dropna().empty:
                regtype_counts = filtered_df['regtypename'].value_counts().reset_index()
                regtype_counts.columns = ['Тип реєстрації', 'Кількість']
                fig_regtype = px.bar(regtype_counts, x='Тип реєстрації', y='Кількість',
                                     title="Учасники за типом реєстрації", color='Тип реєстрації')
                st.plotly_chart(fig_regtype, use_container_width=True)
            else:
                st.info("Немає даних 'regtypename' для розподілу за типом реєстрації.")

        with col2:
            st.subheader("Кількість тестувань за датою (`testdate`)")
            if 'testdate' in filtered_df.columns and not filtered_df['testdate'].dropna().empty:
                # Ensure 'testdate' is datetime
                if pd.api.types.is_datetime64_any_dtype(filtered_df['testdate']):
                    daily_tests = filtered_df.groupby(filtered_df['testdate'].dt.date).size().reset_index(name='Кількість')
                    daily_tests.columns = ['Дата', 'Кількість']
                    fig_daily_tests = px.line(daily_tests, x='Дата', y='Кількість', markers=True, title="Кількість тестувань за днями (для вибраних фільтрів)")
                    st.plotly_chart(fig_daily_tests, use_container_width=True)
                else:
                    st.info("Колонка 'testdate' не є типом datetime. Неможливо згрупувати за датою.")
            else:
                st.info("Немає даних 'testdate' для щоденної динаміки.")

            st.subheader("Розподіл за статтю по роках (фільтровані дані)")
            if 'exam_year' in filtered_df.columns and 'sextypename' in filtered_df.columns and \
               not filtered_df[['exam_year', 'sextypename']].dropna().empty:
                gender_by_year = filtered_df.groupby(['exam_year', 'sextypename']).size().reset_index(name='Кількість')
                if not gender_by_year.empty:
                    fig_gender_year = px.bar(gender_by_year, x='exam_year', y='Кількість', color='sextypename',
                                             barmode='group', title="Розподіл за статтю по роках (для вибраних фільтрів)",
                                             labels={'exam_year':'Рік іспиту', 'sextypename':'Стать'})
                    fig_gender_year.update_xaxes(type='category')
                    st.plotly_chart(fig_gender_year, use_container_width=True)
                else:
                    st.info("Немає даних для розподілу статі по роках.")
            else:
                st.info("Відсутні колонки 'exam_year' або 'sextypename' для розподілу статі по роках.")
                
    # --- 3. Географічний аналіз ---
    with tab3:
        st.header("🗺️ Географічний аналіз")
        # ... (Your existing code for tab3, ensure checks for column existence and empty data) ...
        # Example for one plot in tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Учасники за типом н.п. в розрізі регіонів (`regname`, `settlement_type`)")
            if 'regname' in filtered_df.columns and 'settlement_type' in filtered_df.columns and \
               not filtered_df[['regname', 'settlement_type']].dropna().empty:
                region_settlement_counts = filtered_df.groupby(['regname', 'settlement_type']).size().reset_index(name='Кількість')
                if not region_settlement_counts.empty:
                    fig_region_settlement = px.bar(region_settlement_counts, x='regname', y='Кількість',
                                                   color='settlement_type', title="Розподіл типів н.п. по регіонах",
                                                   labels={'regname':'Регіон', 'settlement_type':'Тип населеного пункту'},
                                                   category_orders={"regname": region_settlement_counts.groupby('regname')['Кількість'].sum().sort_values(ascending=False).index.tolist()})
                    st.plotly_chart(fig_region_settlement, use_container_width=True)
                else:
                    st.info("Немає даних для розподілу типів населених пунктів по регіонах.")
            else:
                st.info("Відсутні колонки 'regname' або 'settlement_type' для цього аналізу.")
        # Add similar checks for other plots in tab3 and tab4

    # --- 4. Аналіз пунктів та закладів ---
    with tab4:
        st.header("🏫 Аналіз пунктів тестування та навчальних закладів")
        # ... (Your existing code for tab4, ensure checks for column existence and empty data) ...

    # --- Sidebar Footer ---
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 Показано дані для **{filtered_df.shape[0]:,}** записів з **{main_df.shape[0]:,}** загальних.")
    st.sidebar.markdown("ℹ️ *Дані виділені з відкритих даних УЦОЯО 2016-2024 років*")
    st.sidebar.markdown("🔗 [Джерело даних](https://testportal.gov.ua/)")