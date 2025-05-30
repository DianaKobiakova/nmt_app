import streamlit as st


main_page = st.Page('main_page.py', title = 'Головна', icon = '🏠')
page_1 = st.Page('page_1.py', title = '🧮 Калькулятор НМТ та Аналіз Шансів на Вступ 🎓', icon = '📄')
page_2 = st.Page('page_2.py', title = '🚀 Аналітичні дашборди на основі даних тестування', icon = '📈')
page_3 = st.Page('analiz.py', title = 'Результати ЗНО/НМТ за фільтрами', icon = '📊')
page_4 = st.Page('page_3.py', title = 'Про нас', icon = '👥')

pg = st.navigation([main_page, page_1, page_2, page_3, page_4])

pg.run()

