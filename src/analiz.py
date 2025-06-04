import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# !!! ВАЖЛИВО: Вкажіть правильний шлях до вашого файлу тут !!!
FILE_PATH = "src/main_df.csv" # Замініть це на реальний шлях

# Функція для завантаження та кешування даних
@st.cache_data
def load_data(file_path):
    if 'dev' in os.environ['ENVIROMENT_MODE']:
        st.warning("Завантаження моделей НМТ вимкнено в режимі розробки. "
                   "Перевірте, чи встановлено змінну оточення ENVIROMENT_MODE у 'prod' для завантаження моделей.")
    elif 'prod' in os.environ['ENVIROMENT_MODE']:
        st.info("Завантаження моделей НМТ увімкнено 'prod'.")
        import boto3
        s3 = boto3.client('s3')
        s3.download_file('nmt', 'main_df.csv', 'src/main_df.csv')

    """Завантажує дані з файлу."""
    if not os.path.exists(file_path):
        st.error(f"Файл не знайдено за шляхом: {file_path}")
        return None
    try:
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        else:
            st.error(f"Непідтримуваний формат файлу: {file_extension}. Будь ласка, вкажіть шлях до CSV або Excel файлу.")
            return None
        return df
    except Exception as e:
        st.error(f"Помилка при читанні файлу '{file_path}': {e}")
        return None

def run_dashboard():
    """Основна функція для запуску дашборду."""
    st.set_page_config(page_title="Дашборд Аналізу Балів ЗНО", layout="wide")
    st.title("📊 Дашборд Аналізу Результатів ЗНО")

    df_original = load_data(FILE_PATH)

    if df_original is None:
        st.error(f"Не вдалося завантажити дані з файлу: {FILE_PATH}. "
                 f"Будь ласка, перевірте правильність шляху у змінній FILE_PATH у коді скрипта "
                 f"та чи доступний файл для читання.")
        st.stop()

    required_columns = ['exam_year', 'regname', 'settlement_type', 'settlement_name', 'eoname',
                        'ukrball100', 'histball100', 'mathball100']
    missing_cols = [col for col in required_columns if col not in df_original.columns]
    if missing_cols:
        st.error(f"У завантаженому файлі відсутні необхідні колонки: {', '.join(missing_cols)}")
        st.stop()

    st.sidebar.header("Фільтри:")

    # 0. Фільтр за роком (exam_year)
    try:
        # Спробуємо перетворити на цілі числа, якщо це можливо (для кращого сортування)
        years = ['Всі роки'] + sorted(df_original['exam_year'].astype(int).unique().tolist())
    except ValueError:
        # Якщо не вдається перетворити на int
        years = ['Всі роки'] + sorted(df_original['exam_year'].astype(str).unique().tolist())
        
    selected_year = st.sidebar.selectbox("Оберіть рік ЗНО:", years)

    if selected_year == 'Всі роки':
        df_after_year = df_original.copy()
    else:
        # Переконуємося, що порівнюємо однакові типи даних
        if isinstance(selected_year, int) and pd.api.types.is_numeric_dtype(df_original['exam_year']):
             df_after_year = df_original[df_original['exam_year'] == selected_year]
        else: # Якщо роки у файлі або у виборі є рядками
             df_after_year = df_original[df_original['exam_year'].astype(str) == str(selected_year)]

    # 1. Фільтр за регіоном (regname)
    regions = ['Всі'] + sorted(df_after_year['regname'].astype(str).unique().tolist())
    selected_region = st.sidebar.selectbox("Оберіть область:", regions)

    if selected_region == 'Всі':
        df_after_region = df_after_year.copy()
    else:
        df_after_region = df_after_year[df_after_year['regname'] == selected_region]

    # 2. Фільтр за типом населеного пункту (settlement_type)
    settlement_types = ['Всі'] + sorted(df_after_region['settlement_type'].astype(str).unique().tolist())
    selected_settlement_type = st.sidebar.selectbox("Оберіть тип населеного пункту:", settlement_types)

    if selected_settlement_type == 'Всі':
        df_after_settlement_type = df_after_region.copy()
    else:
        df_after_settlement_type = df_after_region[df_after_region['settlement_type'] == selected_settlement_type]
    
    # 3. Фільтр за назвою населеного пункту (випадаючий список)
    settlement_names_options = ['Всі'] + sorted(df_after_settlement_type['settlement_name'].astype(str).unique().tolist())
    selected_settlement_name = st.sidebar.selectbox("Оберіть назву населеного пункту:", settlement_names_options)

    if selected_settlement_name == 'Всі':
        df_after_settlement_name = df_after_settlement_type.copy()
    else:
        df_after_settlement_name = df_after_settlement_type[df_after_settlement_type['settlement_name'] == selected_settlement_name]

    # 4. Фільтр за назвою навчального закладу (випадаючий список)
    school_names_options = ['Всі'] + sorted(df_after_settlement_name['eoname'].astype(str).unique().tolist())
    selected_school = st.sidebar.selectbox("Оберіть навчальний заклад (ЗО):", school_names_options)

    if selected_school == 'Всі':
        final_filtered_df = df_after_settlement_name.copy()
    else:
        final_filtered_df = df_after_settlement_name[df_after_settlement_name['eoname'] == selected_school]
    
    # Створення табів
    tab1_title = "📊 Статистика Результатів ЗНО" 
    tab1, = st.tabs([tab1_title])

    with tab1:
        st.header(tab1_title)
        st.markdown("---")
        st.subheader("Результати Фільтрації")

        if final_filtered_df.empty:
            st.warning("За обраними фільтрами дані відсутні.")
        else:
            st.write(f"Знайдено **{len(final_filtered_df)}** записів за вашими критеріями.")

            st.markdown("### 📊 Загальна Статистика за Предметами")
            score_cols = ['ukrball100', 'histball100', 'mathball100']
            
            df_for_stats = final_filtered_df.copy()
            for col in score_cols:
                df_for_stats[col] = pd.to_numeric(df_for_stats[col], errors='coerce')

            stats_df = df_for_stats[score_cols].dropna(subset=score_cols, how='all')

            if stats_df.empty or all(stats_df[col].isnull().all() for col in score_cols):
                st.warning("Немає числових даних для розрахунку статистики балів після очищення.")
            else:
                st.write("Описова статистика для відфільтрованих даних (всі предмети разом):")
                st.dataframe(stats_df.describe().T.rename(columns={
                    'count': 'Кількість', 'mean': 'Середнє', 'std': 'Станд. відхилення',
                    'min': 'Мін.', '25%': '25-й перцентиль', '50%': 'Медіана (50-й перц.)',
                    '75%': '75-й перцентиль', 'max': 'Макс.'
                }))

                st.markdown("---")
                st.subheader("Детальна Статистика та Розподіл по Кожному Предмету")
                
                subject_map = {
                    'ukrball100': '🇺🇦 Українська мова та література',
                    'histball100': '📜 Історія України',
                    'mathball100': '🧮 Математика'
                }
                
                cols_display = st.columns(len(subject_map))

                for i, (col_name, subject_title) in enumerate(subject_map.items()):
                    with cols_display[i]:
                        st.markdown(f"##### {subject_title} (`{col_name}`)")
                        subject_data = stats_df[[col_name]].dropna()
                        if not subject_data.empty:
                            st.write("Статистика:")
                            st.dataframe(subject_data.describe().T.rename(columns={
                                'count': 'Кількість', 'mean': 'Середнє', 'std': 'Станд. відх.',
                                'min': 'Мін.', '25%': 'Q1', '50%': 'Медіана', '75%': 'Q3', 'max': 'Макс.'
                            }), height=150)

                            st.write("Розподіл балів (Гістограма):")
                            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                            ax_hist.hist(subject_data[col_name], bins='auto', edgecolor='black', color='blue')
                            ax_hist.set_xlabel('Бали')
                            ax_hist.set_ylabel('Кількість учнів')
                            ax_hist.grid(axis='y', alpha=0.75)
                            plt.tight_layout()
                            st.pyplot(fig_hist)
                            plt.close(fig_hist)
                        else:
                            st.info("Дані для цього предмету відсутні.")
                
                st.markdown("---")
                st.subheader("Порівняльний Розподіл Балів за Предметами (Бокс-плот)")
                
                plot_data_boxplot = stats_df.dropna(axis=1, how='all') 
                
                if not plot_data_boxplot.empty:
                    fig_box, ax_box = plt.subplots(figsize=(10, 6))
                    plot_data_boxplot.plot(kind='box', ax=ax_box, patch_artist=True)
                    ax_box.set_title('Порівняння розподілу балів за вибраними предметами')
                    ax_box.set_ylabel('Бали')
                    ax_box.set_xticklabels([subject_map.get(col, col) for col in plot_data_boxplot.columns], rotation=0)
                    ax_box.grid(axis='y', alpha=0.75)
                    st.pyplot(fig_box)
                    plt.close(fig_box)
                else:
                    st.info("Недостатньо даних для побудови порівняльного бокс-плоту.")

            st.markdown("---")
            st.subheader("📜 Перегляд Відфільтрованих Даних (перші 100 записів)")
            st.dataframe(final_filtered_df.head(100))

if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-whitegrid')
    run_dashboard()