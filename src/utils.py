def mapping_uk_to_en(df_t):
  df = df_t.copy()
  school_type_mapping = {
    'середня загальноосвітня школа': 'general_school',
    'навчально-виховний комплекс': 'education_complex',
    'ліцей': 'lyceum',
    'спеціалізована школа': 'specialized_school',
    'науковий ліцей': 'science_lyceum',
    'гімназія': 'gymnasium',
    'заклад фахової передвищої освіти': 'pre_higher_institution',
    'заклад вищої освіти': 'higher_education',
    'колегіум': 'collegium',
    'заклад професійної (професійно-технічної) освіти': \
                                          'vocational_institution',
    'загальноосвітня санаторна школа': 'sanatorium_school',
    "навчально-виховне об'єднання": 'education_association',
    'ліцей із посиленою військово-фізичною підготовкою': \
                                          'military_physical_lyceum',
    'спортивний ліцей': 'sports_lyceum',
    'середня загальноосвітня школа-інтернат': 'boarding_general_school',
    'спеціалізована школа-інтернат': 'boarding_special_school',
    'спеціальна загальноосвітня школа': 'special_general_school',
    'колегіум/колеж': 'collegium_college',
    'військовий (військово-морський, військово-спортивний) ліцей': \
                                          'military_lyceum',
    'колеж': 'college',
    'вечірня (змінна) школа': 'evening_school',
    'спеціальна загальноосвітня школа-інтернат': 'special_boarding_school',
    'професійний ліцей відповідного профілю': 'vocational_lyceum',
    'початкова школа': 'primary_school',
    'Пенітенціарна установа': 'penitentiary_institution',
    'мистецький ліцей': 'art_lyceum',
    'спеціальна школа': 'special_school',
    'вищий навчальний заклад III-IV рівнів акредитації': 'higher_edu_lvl_3_4',
    'навчально-реабілітаційний центр': 'rehab_center',
    'школа соціальної реабілітації': 'social_rehab_school',
    'професійний коледж (коледж) спортивного профілю': \
                                          'sports_vocational_college'
  }

  settlement_type_map = {
    'обласний центр': 'regional_center',
    'місто': 'city',
    'село': 'village',
    'смт': 'urban_village',
    'інше': 'other'
  }

  oblast_mapping = {
    'Миколаївська область': 'mykolaiv',
    'Черкаська область': 'cherkasy',
    'Чернігівська область': 'chernihiv',
    'Запорізька область': 'zaporizhzhia',
    'Луганська область': 'luhansk',
    'Рівненська область': 'rivne',
    'Одеська область': 'odesa',
    'Київська область': 'kyiv_region',
    'Вінницька область': 'vinnytsia',
    'Тернопільська область': 'ternopil',
    'Дніпропетровська область': 'dnipropetrovsk',
    'м.Київ': 'kyiv_city',
    'Львівська область': 'lviv',
    'Хмельницька область': 'khmelnytskyi',
    'Харківська область': 'kharkiv',
    'Кіровоградська область': 'kirovohrad',
    'Чернівецька область': 'chernivtsi',
    'Волинська область': 'volyn',
    'Івано-Франківська область': 'ivano_frankivsk',
    'Донецька область': 'donetsk',
    'Полтавська область': 'poltava',
    'Херсонська область': 'kherson',
    'Закарпатська область': 'zakarpattia',
    'Сумська область': 'sumy',
    'Житомирська область': 'zhytomyr'
  }

  df['eotypename'] = df['eotypename'].apply(
      lambda x: school_type_mapping.get(x, 'unknown'))

  df['settlement_type'] = df['settlement_type'].apply(
      lambda x: settlement_type_map.get(x, 'unknown'))

  df['regname'] = df['regname'].apply(
      lambda x: oblast_mapping.get(x, 'unknown'))

  df['sex'] = df['sextypename'].map({'чоловіча': 1, 'жіноча': 0})
  df.drop('sextypename', axis=1, inplace=True)
  df['age'] = df['exam_year'] - df['birth']
  df.drop(columns=['exam_year', 'birth'], inplace=True)
  # regtypename - always starts as 'Випускник' - useless
  # eoname - is the combination of eotypename and settlement_name - useless
  # testdate - useless (when an exam will be)
  # ptregname - region name, less informative than regname
  # settlement_name - name of city/country/villige/town (cannot be informative)
  df.drop(columns=['regtypename', 'eoname', 'testdate',
                   'ptregname', 'settlement_name'], errors='ignore', inplace=True)

  return df


def function_feature_names(transformer, feature_names):
    return ['regname', 'settlement_type', 'eotypename', 'sex', 'age']

