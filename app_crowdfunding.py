# Необходимые библиотеки
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re
import textstat
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pickle
import lightgbm as lgb
import os
import shap
import streamlit.components.v1 as components
import datetime


# --- Page configuration and styles ---
st.set_page_config(
    page_title="Crowdlytics",
    page_icon="logo.svg",
    layout="wide"
)


# Используем st.cache_resource для кэширования модели
@st.cache_resource
def load_model():
    # Проверяем, существует ли файл модели
    if not os.path.exists('lgbm_project_streamlit.pkl'):
        st.error("Файл модели 'lgbm_project_streamlit.pkl' не найден.")
        st.stop()  # Останавливаем выполнение приложения

    # Загружаем модель с помощью pickle
    with open('lgbm_project_streamlit.pkl', 'rb') as f:
        lgbm = pickle.load(f)
    return lgbm

# Вызываем функцию для загрузки модели
lgbm = load_model()

# Загружаем предобученную модель ModernBERT для анализа сентимента
@st.cache_resource
def load_sentiment_model():
    model_name = "clapAI/modernBERT-base-multilingual-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_pipeline

sentiment_pipeline = load_sentiment_model()

# Используем st.cache_resource для кэширования scaler'а
@st.cache_resource
def load_scaler():
    if not os.path.exists('scaler.pkl'):
        st.error("Файл 'scaler.pkl' не найден. Пожалуйста, сохраните обученный scaler.")
        st.stop()
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

# Вызываем функцию для загрузки scaler'а
scaler = load_scaler()

@st.cache_resource
def load_shap_explainer():
    """Loads the pre-trained SHAP explainer from a pickled file."""
    if not os.path.exists('explainer.pkl'):
        st.error("SHAP explainer file 'explainer.pkl' not found.")
        st.stop()
    with open('explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    return explainer

# Call the function to load the explainer
explainer = load_shap_explainer()


if "page" not in st.session_state:
    st.session_state.page = "home"

def navigate_to_input_data():
    st.session_state.page = 'input_data'

def navigate_to_results():
    st.session_state.page = 'results'


# --- Main Page Content ---
if st.session_state.page == "home":
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.image("wide logo.svg", width=170)
        st.markdown('<div style="margin-top: 56px;"></div>', unsafe_allow_html=True)
        st.title("Crowdfunding Project Success Predictor")
        st.subheader("Increase your chances of getting funded on Kickstarter with our forecast model")
        st.markdown('<div style="margin-top: 100px;"></div>', unsafe_allow_html=True)
        st.button("Enter Data & Predict", on_click=navigate_to_input_data)


elif st.session_state.page == "input_data":
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )

    # функции

    # Функция для расчета Fog Index
    def compute_fog_index(text):
        return textstat.gunning_fog(text) if isinstance(text, str) else None

    # Список причинных и соединительных слов (можно дополнить)
    connective_words = [
        'and', 'but', 'because', 'therefore', 'thus', 'so', 'however', 'meanwhile',
        'since', 'although', 'besides', 'furthermore', 'as a result', 'for example',
        'for instance', 'in conclusion', 'in addition', 'on the other hand', 'in fact',
        'due to', 'owing to', 'despite', 'while', 'as well as', 'nevertheless', 'not only',
        'also', 'consequently', 'accordingly', 'though', 'either', 'nor', 'either...or',
        'both...and', 'yet', 'for', 'even though', 'on the contrary', 'in spite of', 'that is',
        'such as', 'this means', 'so that', 'thus', 'hence', 'if', 'unless', 'when', 'whenever',
        'while', 'before', 'after', 'even if', 'as long as', 'provided that', 'as', 'in order that',
        'in case', 'on condition that', 'so as to', 'with the result that', 'by virtue of', 'considering that',
        'meanwhile', 'because of', 'on the grounds that', 'provided', 'given that', 'by the way', 'so far as',
        'according to', 'as well as', 'subsequently', 'therefore', 'likewise', 'similarly', 'correspondingly'
    ]

    # Функция для подсчета логичности текста
    def calculate_logicality(text):
        # Приводим текст к нижнему регистру и разбиваем на слова
        words = re.findall(r'\b\w+\b', text.lower())
        # Считаем количество соединительных слов
        connective_count = sum(1 for word in words if word in connective_words)
        # Общая длина текста
        total_words = len(words)
        # Вычисляем пропорцию логичных слов
        if total_words == 0:
            return 0  # если текст пустой
        else:
            return connective_count / total_words

    #контент
    st.image("wide logo.svg", width=170)

    st.markdown('<div style="margin-top: 56px;"></div>', unsafe_allow_html=True)
    st.title("Project Details")

    st.subheader("Please fill in the details below to receive a prediction*. "
                 "For the most precise prediction, please provide as much information about your project as possible.")

    st.markdown('<div style="margin-top: 56px;"></div>', unsafe_allow_html=True)
    st.header("Basic Characteristics of the Project")

    if 'goal' in st.session_state:
        base_goal = st.session_state['goal']
    else:
        base_goal = 1.00
    goal = st.number_input("Goal (in USD)", value=base_goal, min_value=1.00, format="%.2f",
                           help="How much do you want to raise through crowdfunding?")
    st.session_state.goal = goal

    if 'period' in st.session_state:
        base_period = st.session_state['period']
    else:
        base_period = 1
    period = st.number_input("Campaign Duration (in days)", value=base_period, min_value=1, format="%d",
                           help="How many days will your campaign last?")
    st.session_state.period = period

    if 'year' in st.session_state:
        base_year = st.session_state['year']
    else:
        base_year = datetime.datetime.now().year
    year = st.number_input("Year", value=base_year, min_value=2009, format="%d",
                           help="What year do you want to launch your campaign?")
    st.session_state.year = year

    options_country = ['USA', 'Others']
    country = st.selectbox("Country", options=options_country,
                           help="In which country will your campaign be launched?")
    st.session_state.country = country

    options_category = ['Music', 'Technology', 'Fashion', 'Food', 'Art', 'Publishing',
       'Film & Video', 'Theater', 'Dance', 'Photography', 'Comics',
       'Design', 'Journalism', 'Crafts', 'Games']
    category = st.selectbox("Category", options=options_category,
                            help="What category does your product or service belong to?")
    st.session_state.category = category


    st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)
    st.header("Textual Characteristics of the Project")

    if 'desc' in st.session_state:
        base_desc = st.session_state['desc']
    else:
        base_desc = ''
    desc = st.text_area("Project Description (in English)", value=base_desc,
                  help="Describe your product or service in detail.")
    st.session_state.desc = desc

    if 'risks' in st.session_state:
        base_risks = st.session_state['risks']
    else:
        base_risks = ''
    risks = st.text_area("Risks Description (in English)", value=base_risks,
                  help="Describe the possible risks of the project implementation.")
    st.session_state.risks = risks


    def calculate_and_save_prediction():
        # Преобразование данных

        # Description
        description_length = len(re.findall(r'\b\w+\b', st.session_state.desc))
        st.session_state.description_length = description_length
        fog_index_description = compute_fog_index(st.session_state.desc)
        st.session_state.fog_index_description = fog_index_description
        logicality_description = calculate_logicality(st.session_state.desc)
        st.session_state.logicality_description = logicality_description
        # Функция для анализа сентимента
        def analyze_sentiment(text):
            return sentiment_pipeline(text, truncation=True, max_length=4096)[0]["label"]
        # Анализируем сентимент для каждого текста
        sentiment_description = analyze_sentiment(st.session_state.desc)
        st.session_state.sentiment_description = sentiment_description

        # Risks
        risks_length = len(re.findall(r'\b\w+\b', st.session_state.risks))
        st.session_state.risks_length = risks_length
        fog_index_risks = compute_fog_index(st.session_state.risks)
        st.session_state.fog_index_risks = fog_index_risks
        logicality_risks = calculate_logicality(st.session_state.risks)
        st.session_state.logicality_risks = logicality_risks

        # логарифмирование
        # Логарифмируем те переменные, которые не имеют нулевых значений
        goal_log = np.log(st.session_state.goal)
        st.session_state.goal_log = goal_log
        period_log = np.log(st.session_state.period)
        st.session_state.period_log = period_log

        # Могут быть нули
        description_length_log = np.log(
            st.session_state.description_length + 1) if st.session_state.description_length == 0 else np.log(
                st.session_state.description_length)
        st.session_state.description_length_log = description_length_log
        risks_length_log = np.log(
            st.session_state.risks_length + 1) if st.session_state.risks_length == 0 else np.log(
                st.session_state.risks_length)
        st.session_state.risks_length_log = risks_length_log
        fog_index_description_log = np.log(
            st.session_state.fog_index_description + 1) if st.session_state.fog_index_description == 0 else np.log(
                st.session_state.fog_index_description)
        st.session_state.fog_index_description_log = fog_index_description_log
        fog_index_risks_log = np.log(
            st.session_state.fog_index_risks + 1) if st.session_state.fog_index_risks == 0 else np.log(
                st.session_state.fog_index_risks)
        st.session_state.fog_index_risks_log = fog_index_risks_log
        logicality_description_log = np.log(
            st.session_state.logicality_description + 1) if st.session_state.logicality_description == 0 else np.log(
                st.session_state.logicality_description)
        st.session_state.logicality_description_log = logicality_description_log
        logicality_risks_log = np.log(
            st.session_state.logicality_risks + 1) if st.session_state.logicality_risks == 0 else np.log(
                st.session_state.logicality_risks)
        st.session_state.logicality_risks_log = logicality_risks_log

        # обработка категориальных переменных

        # сентимент
        if sentiment_description == "neutral":
            sentiment_description_neutral = 1
            sentiment_description_positive = 0
        elif sentiment_description == "positive":
            sentiment_description_positive = 1
            sentiment_description_neutral = 0
        else:
            sentiment_description_neutral = 0
            sentiment_description_positive = 0

        st.session_state.sentiment_description_neutral = sentiment_description_neutral
        st.session_state.sentiment_description_positive = sentiment_description_positive

        # страна
        if country == 'USA':
            USA = 1
        else:
            USA = 0

        st.session_state.USA = USA

        # категория
        st.session_state.category_new_Comics = 1 if st.session_state.category == 'Comics' else 0
        st.session_state.category_new_Crafts = 1 if st.session_state.category == 'Crafts' else 0
        st.session_state.category_new_Dance = 1 if st.session_state.category == 'Dance' else 0
        st.session_state.category_new_Design = 1 if st.session_state.category == 'Design' else 0
        st.session_state.category_new_Fashion = 1 if st.session_state.category == 'Fashion' else 0
        st.session_state.category_new_Film_Video = 1 if st.session_state.category == 'Film & Video' else 0
        st.session_state.category_new_Food = 1 if st.session_state.category == 'Food' else 0
        st.session_state.category_new_Games = 1 if st.session_state.category == 'Games' else 0
        st.session_state.category_new_Journalism = 1 if st.session_state.category == 'Journalism' else 0
        st.session_state.category_new_Music = 1 if st.session_state.category == 'Music' else 0
        st.session_state.category_new_Photography = 1 if st.session_state.category == 'Photography' else 0
        st.session_state.category_new_Publishing = 1 if st.session_state.category == 'Publishing' else 0
        st.session_state.category_new_Technology = 1 if st.session_state.category == 'Technology' else 0
        st.session_state.category_new_Theater = 1 if st.session_state.category == 'Theater' else 0

        # Create a dictionary with the data from st.session_state
        data_abs = {
            'goal': [st.session_state.goal],
            'year': [st.session_state.year],
            'USA': [st.session_state.USA],
            'period': [st.session_state.period],
            'category_new_Comics': [st.session_state.category_new_Comics],
            'category_new_Crafts': [st.session_state.category_new_Crafts],
            'category_new_Dance': [st.session_state.category_new_Dance],
            'category_new_Design': [st.session_state.category_new_Design],
            'category_new_Fashion': [st.session_state.category_new_Fashion],
            'category_new_Film & Video': [st.session_state.category_new_Film_Video],
            'category_new_Food': [st.session_state.category_new_Food],
            'category_new_Games': [st.session_state.category_new_Games],
            'category_new_Journalism': [st.session_state.category_new_Journalism],
            'category_new_Music': [st.session_state.category_new_Music],
            'category_new_Photography': [st.session_state.category_new_Photography],
            'category_new_Publishing': [st.session_state.category_new_Publishing],
            'category_new_Technology': [st.session_state.category_new_Technology],
            'category_new_Theater': [st.session_state.category_new_Theater],
            'description_length': [st.session_state.description_length],
            'risks_length': [st.session_state.risks_length],
            'fog_index_description': [st.session_state.fog_index_description],
            'fog_index_risks': [st.session_state.fog_index_risks],
            'logicality_description': [st.session_state.logicality_description],
            'logicality_risks': [st.session_state.logicality_risks],
            'sentiment_description_neutral': [st.session_state.sentiment_description_neutral],
            'sentiment_description_positive': [st.session_state.sentiment_description_positive]
        }

        # Create a DataFrame from the dictionary
        Xabs = pd.DataFrame(data_abs)
        st.session_state.Xabs = Xabs


        # Create a dictionary with the data from st.session_state
        data = {
            'goal_log': [st.session_state.goal_log],
            'year': [st.session_state.year],
            'USA': [st.session_state.USA],
            'period_log': [st.session_state.period_log],
            'category_new_Comics': [st.session_state.category_new_Comics],
            'category_new_Crafts': [st.session_state.category_new_Crafts],
            'category_new_Dance': [st.session_state.category_new_Dance],
            'category_new_Design': [st.session_state.category_new_Design],
            'category_new_Fashion': [st.session_state.category_new_Fashion],
            'category_new_Film & Video': [st.session_state.category_new_Film_Video],
            'category_new_Food': [st.session_state.category_new_Food],
            'category_new_Games': [st.session_state.category_new_Games],
            'category_new_Journalism': [st.session_state.category_new_Journalism],
            'category_new_Music': [st.session_state.category_new_Music],
            'category_new_Photography': [st.session_state.category_new_Photography],
            'category_new_Publishing': [st.session_state.category_new_Publishing],
            'category_new_Technology': [st.session_state.category_new_Technology],
            'category_new_Theater': [st.session_state.category_new_Theater],
            'description_length_log': [st.session_state.description_length_log],
            'risks_length_log': [st.session_state.risks_length_log],
            'fog_index_description_log': [st.session_state.fog_index_description_log],
            'fog_index_risks_log': [st.session_state.fog_index_risks_log],
            'logicality_description_log': [st.session_state.logicality_description_log],
            'logicality_risks_log': [st.session_state.logicality_risks_log],
            'sentiment_description_neutral': [st.session_state.sentiment_description_neutral],
            'sentiment_description_positive': [st.session_state.sentiment_description_positive]
        }


        # Create a DataFrame from the dictionary
        X = pd.DataFrame(data)
        st.session_state.X = X

        # Масштабирование числовых признаков
        X_scaled = scaler.transform(X)
        st.session_state.X_scaled = X_scaled

        # Получение прогноза
        predicted_proba = lgbm.predict_proba(st.session_state.X_scaled)[0, 1]
        st.session_state.predicted_proba = predicted_proba

        # Переход на страницу с результатами
        navigate_to_results()

    st.markdown('<div style="margin-top: 56px;"></div>', unsafe_allow_html=True)

    st.button("Predict", on_click=calculate_and_save_prediction)
    st.caption("*Your data is used solely for the purpose of generating a prediction and is not stored")

elif st.session_state.page == "results":
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )

    st.image("wide logo.svg", width=170)

    st.markdown('<div style="margin-top: 56px;"></div>', unsafe_allow_html=True)
    st.title("Crowdfunding Success Forecast")

    st.subheader("Here, you can get acquainted with your project's predicted probability of success and explore "
                 "recommendations for running a crowdfunding campaign, generated from the model's insights. "
                 "Please note that this is a predictive model, it may be wrong. "
                 "The model's prediction should be used as a supplementary tool to aid your decision-making, "
                 "not as a definitive guarantee.")

    success_rates = {'Art': {'Others': 40.98, 'USA': 45.1}, 'Comics': {'Others': 70.06, 'USA': 70.11},
                     'Crafts': {'Others': 33.89, 'USA': 26.7}, 'Dance': {'Others': 44.12, 'USA': 64.75},
                     'Design': {'Others': 44.04, 'USA': 38.41}, 'Fashion': {'Others': 47.56, 'USA': 44.22},
                     'Film & Video': {'Others': 49.83, 'USA': 45.93}, 'Food': {'Others': 24.79, 'USA': 25.63},
                     'Games': {'Others': 28.68, 'USA': 29.0}, 'Journalism': {'Others': 25.7, 'USA': 29.13},
                     'Music': {'Others': 52.06, 'USA': 58.45}, 'Photography': {'Others': 35.71, 'USA': 33.9},
                     'Publishing': {'Others': 57.6, 'USA': 53.82}, 'Technology': {'Others': 28.86, 'USA': 34.7},
                     'Theater': {'Others': 58.51, 'USA': 49.61}}

    # Форматируем число и применяем стили через HTML
    formatted_probability = f"{st.session_state.predicted_proba * 100:.2f}%"

    # Используем st.markdown с HTML и CSS для стилизации
    st.markdown(
        f'<div style="font-size: 300px; font-weight: bold; color: #17D989;">{formatted_probability}</div>',
        unsafe_allow_html=True
    )

    if st.session_state.predicted_proba * 100 > success_rates[st.session_state.category][st.session_state.country]:
        st.write(f"the probability of success of your project. "
                 f"This is higher than for similar projects "
                 f"({success_rates[st.session_state.category][st.session_state.country]}%).")
    elif st.session_state.predicted_proba * 100 < success_rates[st.session_state.category][st.session_state.country]:
        st.write(f"the probability of success of your project. "
                 f"This is lower than for similar projects "
                 f"({success_rates[st.session_state.category][st.session_state.country]}%).")
    else:
        st.write(f"the probability of success of your project. "
                 f"This is the same as for similar projects "
                 f"({success_rates[st.session_state.category][st.session_state.country]}%).")

    st.markdown('<div style="margin-top: 120px;"></div>', unsafe_allow_html=True)

    st.header("Recommendations")

    # Get SHAP values for the selected observation
    shap_values_instance = explainer(st.session_state.Xabs)
    shap_values_data = shap_values_instance[0]
    feature_names = shap_values_data.feature_names
    values = shap_values_data.values
    base_value = shap_values_data.base_values
    expected_value = shap_values_data.base_values

    # Your custom feature names
    custom_feature_names = ["Goal", "Year", "Country", "Campaign Duration", "Category 'Comics'", "Category 'Crafts'",
                            "Category 'Dance'",
                            "Category 'Design'", "Category 'Fashion'", "Category 'Film & Video'", "Category 'Food'",
                            "Category 'Games'", "Category 'Journalism'",
                            "Category 'Music'", "Category 'Photography'", "Category 'Publishing'",
                            "Category 'Technology'", "Category 'Theatre'", "Length of the project description",
                            "Length of the project risks",
                            "Readability score of the project description", "Readability score of the project risks",
                            "Logicality of the project description", "Logicality of the project risks",
                            "Neutral sentiment of the project description",
                            "Positive sentiment of the project description"]

    num_features = ["Goal", "Campaign Duration", "Length of the project description",
                            "Length of the project risks",
                            "Readability score of the project description", "Readability score of the project risks",
                            "Logicality of the project description", "Logicality of the project risks"]

    # Replace standard names with custom ones
    original_feature_names = st.session_state.Xabs.columns.tolist()
    name_mapping = dict(zip(original_feature_names, custom_feature_names))
    mapped_feature_names = np.array([name_mapping.get(name, name) for name in feature_names])
    # Create a reverse mapping to find the original names
    reverse_name_mapping = {v: k for k, v in name_mapping.items()}

    # Create a boolean mask for features that contain 'Category', 'Year' or 'Country'
    is_category_or_fixed = np.array(
        ['Category' in name or name in ['Year', 'Country'] for name in mapped_feature_names])

    # Separate the features that can be influenced
    variable_features = mapped_feature_names[~is_category_or_fixed]
    variable_values = values[~is_category_or_fixed]
    original_variable_features = np.array(feature_names)[~is_category_or_fixed]

    # Sort by absolute value
    sorted_indices = np.argsort(np.abs(variable_values))[::-1]
    sorted_values = variable_values[sorted_indices]
    sorted_features = variable_features[sorted_indices]
    sorted_original_features = original_variable_features[sorted_indices]

    # Split into top 9 and the rest
    top_n = 9
    top_values = sorted_values[:top_n]
    top_features = sorted_features[:top_n]
    top_original_features = sorted_original_features[:top_n]
    other_values_to_sum = sorted_values[top_n:]
    other_original_features = original_variable_features[top_n:]

    # Sum the values for the "Other" group
    other_sum = np.sum(values[is_category_or_fixed]) + np.sum(other_values_to_sum)
    other_count = len(values[is_category_or_fixed]) + len(other_values_to_sum)

    # Assemble the final lists, ensuring "Other" is at the end
    final_features = top_features.tolist()
    final_values = top_values.tolist()
    final_original_features = top_original_features.tolist()
    final_features.append(f"Other {other_count} features")
    final_values.append(other_sum)
    final_original_features.append('Other')


    recommendations_dict = {
        "Goal": {
            "increase": "We recommend increasing the goal to further boost your project's chances of success. "
                        "By increasing the goal, you're showing investors you believe in your project in a big way! "
                        "This signals great ambition and can inspire them to invest more generously.",
            "decrease": "We recommend decreasing the goal to further boost your project's chances of success. "
                        "An overly high goal can scare off investors, especially beginners. "
                        "If the amount seems unattainable, they might just pass by. "
                        "It's better to set a realistic goal to increase your chances of success."
        },
        "Campaign Duration": {
            "increase": "We recommend increasing the campaign duration to further boost your project's chances of success. "
                        "You have time! A longer campaign gives you more chances to get your project noticed. "
                        "It's like a marathon, not a sprint—the longer you're in the game, the more people you can attract.",
            "decrease": "We recommend decreasing the campaign duration to further boost your project's chances of success. "
                        "No one likes to wait. An overly long campaign might seem drawn-out and uncertain to investors. "
                        "Make it shorter to create a sense of urgency and encourage people to act faster."
        },
        "Length of the project description": {
            "increase": "We recommend increasing the description length to further boost your project's chances of success. "
                        "Investors want to know what they're investing in! "
                        "A detailed and well-thought-out description shows you're serious about the project. "
                        "Don't be afraid to share details — they help build trust.",
            "decrease": "We recommend decreasing the description length to further boost your project's chances of success. "
                        "Don't overload them! Too much text can be confusing and overwhelming. "
                        "Focus on what's most important and remove anything unnecessary. "
                        "Brevity is the soul of wit, especially in a project description."
        },
        "Length of the project risks": {
            "increase": "We recommend increasing the risks description length to further boost your project's chances of success. "
                        "Professionals value honesty. By openly describing risks, "
                        "you show your competence and readiness for any challenges. "
                        "This demonstrates that you've thought everything through and makes your project seem more reliable.",
            "decrease": "We recommend decreasing the risks description length to further boost your project's chances of success. "
                        "Be careful! Too many risks can make investors think the project is too dangerous. "
                        "Choose the most important ones and describe them clearly, without causing panic."
        },
        "Readability score of the project description": {
            "increase": "We recommend increasing the readability score to further boost your project's chances of success. "
                        "Use more professional language to show you're an expert in your field. "
                        "This commands respect and trust, which is very important for investors.",
            "decrease": "We recommend decreasing the readability score to further boost your project's chances of success. "
                        "Don't overcomplicate it! If your text is too complex, investors might not even understand your idea. "
                        "Make it simple and clear to reach a wider audience."
        },
        "Readability score of the project risks": {
            "increase": "We recommend increasing the risks readability score to further boost your project's chances of success. "
                        "Use more professional language to show you're an expert in your field. "
                        "This commands respect and trust, which is very important for investors.",
            "decrease": "We recommend decreasing the risks readability score to further boost your project's chances of success. "
                        "Don't overcomplicate it! If your text is too complex, investors might not even understand your idea. "
                        "Make it simple and clear to reach a wider audience."
        },
        "Logicality of the project description": {
            "increase": "We recommend increasing the description logicality to further boost your project's chances of success. "
                        "Your text seems a bit confusing. "
                        "Use more connecting words and logical structures so ideas flow smoothly. "
                        "This will help investors follow your train of thought more easily.",
            "decrease": "We recommend decreasing the description logicality to further boost your project's chances of success. "
                        "Your text has too many 'but', 'however', 'therefore'. This makes it heavy to read. "
                        "Simplify the structure so the text is smoother and easier to understand."
        },
        "Logicality of the project risks": {
            "increase": "We recommend increasing the risks logicality to further boost your project's chances of success. "
                        "Your text seems a bit confusing. "
                        "Use more connecting words and logical structures so ideas flow smoothly. "
                        "This will help investors follow your train of thought more easily.",
            "decrease": "We recommend decreasing the risks logicality to further boost your project's chances of success. "
                        "Your text has too many 'but', 'however', 'therefore'. This makes it heavy to read. "
                        "Simplify the structure so the text is smoother and easier to understand."
        },
        "Neutral sentiment of the project description": {
            "increase": "Add some positivity and friendliness to your description! "
                        "Investors are more likely to fund projects that evoke positive emotions and seem more human.",
            "decrease": "Your text is too emotional. Investors are looking for professionalism. "
                        "Try to make the tone more restrained and neutral so you don't seem naive."
        },
        "Positive sentiment of the project description": {
            "increase": "Add some positivity and friendliness to your description! "
                        "Investors are more likely to fund projects that evoke positive emotions and seem more human.",
            "decrease": "Your text is too emotional. Investors are looking for professionalism. "
                        "Try to make the tone more restrained and neutral so you don't seem naive."
        },
    }

    # Add units for clarity
    units_dict = {
        "Goal": "USD",
        "Campaign Duration": "days",
        "Length of the project description": "words",
        "Length of the project risks": "words",
        "Readability score of the project description": "points",
        "Readability score of the project risks": "points",
        "Logicality of the project description": "points",
        "Logicality of the project risks": "points",
        "Neutral sentiment of the project description": "points",
        "Positive sentiment of the project description": "points",
    }

    # Separate list for binary features
    binary_features = [
        "Positive sentiment of the project description",
        "Neutral sentiment of the project description"
    ]

    optimal_range = {('Art', 'Others'): {'Goal': {'lower': 150.00000000000009, 'upper': 2999.9999999999977},
                                         'Campaign Duration': {'lower': 12.100000000000009, 'upper': 34.99999999999998},
                                         'Length of the project description': {'lower': 349.7999999999999, 'upper': 1309.2000000000003},
                                         'Length of the project risks': {'lower': 78.38799999999998, 'upper': 226.58800000000022},
                                         'Readability score of the project description': {'lower': 9.553, 'upper': 14.247000000000002},
                                         'Readability score of the project risks': {'lower': 16.417719197707736, 'upper': 21.735719197707738},
                                         'Logicality of the project description': {'lower': 0.0777191219528276, 'upper': 0.1157600969304808},
                                         'Logicality of the project risks': {'lower': 0.08741878767114333, 'upper': 0.13815546580445934},
                                         'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                         'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Art', 'USA'): {'Goal': {'lower': 249.9999999999999, 'upper': 3499.9999999999995},
                                      'Campaign Duration': {'lower': 13.999999999999996, 'upper': 36.0},
                                      'Length of the project description': {'lower': 337.5, 'upper': 1110.0},
                                      'Length of the project risks': {'lower': 79.488, 'upper': 228.48799999999991},
                                      'Readability score of the project description': {'lower': 9.74, 'upper': 14.25},
                                      'Readability score of the project risks': {'lower': 16.504719197707736, 'upper': 21.778719197707737},
                                      'Logicality of the project description': {'lower': 0.08609722879856183, 'upper': 0.1170338695827429},
                                      'Logicality of the project risks': {'lower': 0.0933049858732697, 'upper': 0.13970826083551524},
                                      'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 2.718281828459045},
                                      'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Comics', 'Others'): {'Goal': {'lower': 253.99999999999994, 'upper': 3539.600000000003},
                                            'Campaign Duration': {'lower': 17.999999999999996, 'upper': 32.0},
                                            'Length of the project description': {'lower': 396.1, 'upper': 1570.8},
                                            'Length of the project risks': {'lower': 73.48800000000003, 'upper': 207.48800000000003},
                                            'Readability score of the project description': {'lower': 9.828999999999999, 'upper': 13.391},
                                            'Readability score of the project risks': {'lower': 16.029719197707735, 'upper': 21.103719197707736},
                                            'Logicality of the project description': {'lower': 0.07953048275174344, 'upper': 0.1152648636167586},
                                            'Logicality of the project risks': {'lower': 0.08983613807336693, 'upper': 0.14667194932094285},
                                            'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                            'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Comics', 'USA'): {'Goal': {'lower': 306.60000000000025, 'upper': 4170.000000000005},
                                         'Campaign Duration': {'lower': 16.700000000000003, 'upper': 31.0},
                                         'Length of the project description': {'lower': 374.79999999999995, 'upper': 1433.1000000000004},
                                         'Length of the project risks': {'lower': 72.48800000000003, 'upper': 207.488},
                                         'Readability score of the project description': {'lower': 9.400000000000002, 'upper': 13.099999999999998},
                                         'Readability score of the project risks': {'lower': 14.619719197707733, 'upper': 21.28171919770774},
                                         'Logicality of the project description': {'lower': 0.08627421264763181, 'upper': 0.11604629280645454},
                                         'Logicality of the project risks': {'lower': 0.09455135749446864, 'upper': 0.14812972577871936},
                                         'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                         'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Crafts', 'Others'): {'Goal': {'lower': 194.0, 'upper': 2999.9999999999977},
                                            'Campaign Duration': {'lower': 13.999999999999996, 'upper': 38.99999999999999},
                                            'Length of the project description': {'lower': 350.9999999999999, 'upper': 1337.9999999999995},
                                            'Length of the project risks': {'lower': 76.48800000000003, 'upper': 222.58799999999994},
                                            'Readability score of the project description': {'lower': 9.657, 'upper': 13.679999999999998},
                                            'Readability score of the project risks': {'lower': 17.274719197707736, 'upper': 22.329719197707735},
                                            'Logicality of the project description': {'lower': 0.07845127882756406, 'upper': 0.1140088308138319},
                                            'Logicality of the project risks': {'lower': 0.09101260866160223, 'upper': 0.14695463764710945},
                                            'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 2.718281828459045},
                                            'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Crafts', 'USA'): {'Goal': {'lower': 199.99999999999991, 'upper': 2999.9999999999977},
                                         'Campaign Duration': {'lower': 13.999999999999996, 'upper': 36.0},
                                         'Length of the project description': {'lower': 357.19999999999993, 'upper': 1220.0000000000002},
                                         'Length of the project risks': {'lower': 80.48799999999997, 'upper': 223.48799999999994},
                                         'Readability score of the project description': {'lower': 9.650000000000002, 'upper': 13.475000000000001},
                                         'Readability score of the project risks': {'lower': 17.361719197707732, 'upper': 22.09371919770774},
                                         'Logicality of the project description': {'lower': 0.08478751970111835, 'upper': 0.11514744440395915},
                                         'Logicality of the project risks': {'lower': 0.09351260866160223, 'upper': 0.14131071164263218},
                                         'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 2.718281828459045},
                                         'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Dance', 'Others'): {'Goal': {'lower': 499.99999999999983, 'upper': 3999.9999999999995},
                                           'Campaign Duration': {'lower': 15.0, 'upper': 39.69999999999999},
                                           'Length of the project description': {'lower': 325.8, 'upper': 1119.0000000000007},
                                           'Length of the project risks': {'lower': 84.48799999999999, 'upper': 212.08800000000014},
                                           'Readability score of the project description': {'lower': 11.330000000000002, 'upper': 15.769},
                                           'Readability score of the project risks': {'lower': 16.005719197707737, 'upper': 22.27771919770774},
                                           'Logicality of the project description': {'lower': 0.07642625177128698, 'upper': 0.10747647457598054},
                                           'Logicality of the project risks': {'lower': 0.05296674148530021, 'upper': 0.10975753827267351},
                                           'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 2.718281828459045},
                                           'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Dance', 'USA'): {'Goal': {'lower': 699.9999999999998, 'upper': 5999.999999999995},
                                        'Campaign Duration': {'lower': 16.6, 'upper': 36.99999999999998},
                                        'Length of the project description': {'lower': 302.09999999999997, 'upper': 990.9000000000002},
                                        'Length of the project risks': {'lower': 84.48799999999999, 'upper': 229.988},
                                        'Readability score of the project description': {'lower': 11.330000000000002, 'upper': 16.04},
                                        'Readability score of the project risks': {'lower': 17.105719197707742, 'upper': 21.96971919770774},
                                        'Logicality of the project description': {'lower': 0.08449515585449201, 'upper': 0.11307609099501185},
                                        'Logicality of the project risks': {'lower': 0.07085574591650422, 'upper': 0.12784124620073387},
                                        'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 2.718281828459045},
                                        'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Design', 'Others'): {'Goal': {'lower': 224.20000000000002, 'upper': 2999.9999999999977},
                                            'Campaign Duration': {'lower': 13.999999999999996, 'upper': 36.60000000000001},
                                            'Length of the project description': {'lower': 358.1999999999999, 'upper': 1554.400000000001},
                                            'Length of the project risks': {'lower': 75.48799999999999, 'upper': 209.988},
                                            'Readability score of the project description': {'lower': 9.759, 'upper': 13.491999999999999},
                                            'Readability score of the project risks': {'lower': 16.435719197707733, 'upper': 21.82071919770774},
                                            'Logicality of the project description': {'lower': 0.07832062160599775, 'upper': 0.1133020218489021},
                                            'Logicality of the project risks': {'lower': 0.08983613807336693, 'upper': 0.13912071676971036},
                                            'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                            'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Design', 'USA'): {'Goal': {'lower': 299.99999999999994, 'upper': 3999.9999999999995},
                                         'Campaign Duration': {'lower': 13.999999999999996, 'upper': 34.99999999999998},
                                         'Length of the project description': {'lower': 363.99999999999994, 'upper': 1199.7},
                                         'Length of the project risks': {'lower': 78.48799999999997, 'upper': 232.48799999999991},
                                         'Readability score of the project description': {'lower': 9.651000000000002, 'upper': 13.947999999999999},
                                         'Readability score of the project risks': {'lower': 16.337719197707735, 'upper': 21.401719197707735},
                                         'Logicality of the project description': {'lower': 0.08444491784161129, 'upper': 0.1140481369107108},
                                         'Logicality of the project risks': {'lower': 0.09198202800046563, 'upper': 0.1372261277728807},
                                         'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 2.718281828459045},
                                         'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Fashion', 'Others'): {'Goal': {'lower': 199.99999999999991, 'upper': 3230.0000000000005},
                                             'Campaign Duration': {'lower': 13.999999999999996, 'upper': 38.99999999999999},
                                             'Length of the project description': {'lower': 374.0000000000001, 'upper': 1350.1999999999998},
                                             'Length of the project risks': {'lower': 83.488, 'upper': 249.58800000000008},
                                             'Readability score of the project description': {'lower': 8.242999999999999, 'upper': 11.830000000000002},
                                             'Readability score of the project risks': {'lower': 16.545719197707736, 'upper': 21.581719197707734},
                                             'Logicality of the project description': {'lower': 0.08058381296009819, 'upper': 0.11644870764536573},
                                             'Logicality of the project risks': {'lower': 0.08656816421715775, 'upper': 0.13018616238061048},
                                             'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                             'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Fashion', 'USA'): {'Goal': {'lower': 259.99999999999994, 'upper': 2999.9999999999977},
                                          'Campaign Duration': {'lower': 13.999999999999996, 'upper': 34.99999999999998},
                                          'Length of the project description': {'lower': 360.99999999999983, 'upper': 1153.2000000000003},
                                          'Length of the project risks': {'lower': 84.48799999999999, 'upper': 235.48800000000008},
                                          'Readability score of the project description': {'lower': 8.528999999999998, 'upper': 11.951000000000002},
                                          'Readability score of the project risks': {'lower': 16.681719197707736, 'upper': 21.62171919770773},
                                          'Logicality of the project description': {'lower': 0.084834547877978, 'upper': 0.11719223200585156},
                                          'Logicality of the project risks': {'lower': 0.08815546580445933, 'upper': 0.13549022060190075},
                                          'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                          'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Film & Video', 'Others'): {'Goal': {'lower': 249.9999999999999, 'upper': 3557.4999999999995},
                                                  'Campaign Duration': {'lower': 15.999999999999998, 'upper': 39.99999999999998},
                                                  'Length of the project description': {'lower': 378.0000000000001, 'upper': 1708.0000000000007},
                                                  'Length of the project risks': {'lower': 78.48799999999997, 'upper': 244.5880000000002},
                                                  'Readability score of the project description': {'lower': 10.343, 'upper': 14.65},
                                                  'Readability score of the project risks': {'lower': 17.201719197707735, 'upper': 22.441719197707734},
                                                  'Logicality of the project description': {'lower': 0.07438033901468878, 'upper': 0.11776021686352567},
                                                  'Logicality of the project risks': {'lower': 0.08506666271565623, 'upper': 0.13305342498813286},
                                                  'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                                  'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Film & Video', 'USA'): {'Goal': {'lower': 499.99999999999983, 'upper': 4500.000000000001},
                                               'Campaign Duration': {'lower': 17.0, 'upper': 36.0},
                                               'Length of the project description': {'lower': 365.9999999999999, 'upper': 1500.9999999999998},
                                               'Length of the project risks': {'lower': 80.48799999999997, 'upper': 240.48800000000006},
                                               'Readability score of the project description': {'lower': 10.160000000000002, 'upper': 14.370000000000001},
                                               'Readability score of the project risks': {'lower': 17.01571919770774, 'upper': 22.32971919770774},
                                               'Logicality of the project description': {'lower': 0.08560053536812497, 'upper': 0.1179240237696968},
                                               'Logicality of the project risks': {'lower': 0.09188217387899357, 'upper': 0.13536043474855874},
                                               'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                               'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Food', 'Others'): {'Goal': {'lower': 199.99999999999991, 'upper': 3499.9999999999995},
                                          'Campaign Duration': {'lower': 15.0, 'upper': 36.99999999999998},
                                          'Length of the project description': {'lower': 372.4, 'upper': 1363.8000000000002},
                                          'Length of the project risks': {'lower': 84.18799999999999, 'upper': 258.488},
                                          'Readability score of the project description': {'lower': 9.92, 'upper': 14.2},
                                          'Readability score of the project risks': {'lower': 16.93171919770774, 'upper': 21.876719197707736},
                                          'Logicality of the project description': {'lower': 0.08272194437800673, 'upper': 0.11697423014617764},
                                          'Logicality of the project risks': {'lower': 0.08656816421715775, 'upper': 0.13627576655633905},
                                          'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                          'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Food', 'USA'): {'Goal': {'lower': 199.99999999999991, 'upper': 3999.9999999999995},
                                       'Campaign Duration': {'lower': 18.999999999999996, 'upper': 34.99999999999998},
                                       'Length of the project description': {'lower': 362.9999999999999, 'upper': 1164.9999999999998},
                                       'Length of the project risks': {'lower': 85.48799999999997, 'upper': 248.48800000000003},
                                       'Readability score of the project description': {'lower': 9.88, 'upper': 13.83},
                                       'Readability score of the project risks': {'lower': 16.99171919770774, 'upper': 21.94171919770773},
                                       'Logicality of the project description': {'lower': 0.08782119922974257, 'upper': 0.11726678316907997},
                                       'Logicality of the project risks': {'lower': 0.09161866926766284, 'upper': 0.13634989554077206},
                                       'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 2.718281828459045},
                                       'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Games', 'Others'): {'Goal': {'lower': 199.99999999999991, 'upper': 3999.9999999999995},
                                           'Campaign Duration': {'lower': 13.999999999999996, 'upper': 36.99999999999998},
                                           'Length of the project description': {'lower': 393.2, 'upper': 1653.3999999999996},
                                           'Length of the project risks': {'lower': 84.48799999999999, 'upper': 238.088},
                                           'Readability score of the project description': {'lower': 9.79, 'upper': 13.450000000000001},
                                           'Readability score of the project risks': {'lower': 16.011719197707734, 'upper': 21.301719197707733},
                                           'Logicality of the project description': {'lower': 0.08404015191869393, 'upper': 0.11552906132609644},
                                           'Logicality of the project risks': {'lower': 0.08656816421715775, 'upper': 0.12883869561812394},
                                           'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                           'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Games', 'USA'): {'Goal': {'lower': 199.99999999999991, 'upper': 5000.000000000004},
                                        'Campaign Duration': {'lower': 15.0, 'upper': 34.99999999999998},
                                        'Length of the project description': {'lower': 376.00000000000006, 'upper': 1553.6},
                                        'Length of the project risks': {'lower': 85.48799999999997, 'upper': 245.48799999999991},
                                        'Readability score of the project description': {'lower': 9.699999999999998, 'upper': 13.328999999999999},
                                        'Readability score of the project risks': {'lower': 16.421719197707734, 'upper': 21.395719197707734},
                                        'Logicality of the project description': {'lower': 0.08605606079549025, 'upper': 0.11742194016098637},
                                        'Logicality of the project risks': {'lower': 0.09072401839036702, 'upper': 0.1326127089289819},
                                        'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                        'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Journalism', 'Others'): {'Goal': {'lower': 199.99999999999991, 'upper': 3079.9999999999995},
                                                'Campaign Duration': {'lower': 14.199999999999998, 'upper': 39.99999999999998},
                                                'Length of the project description': {'lower': 353.69999999999993, 'upper': 1200.6},
                                                'Length of the project risks': {'lower': 85.08799999999998, 'upper': 224.488},
                                                'Readability score of the project description': {'lower': 10.715, 'upper': 14.97},
                                                'Readability score of the project risks': {'lower': 13.731719197707735, 'upper': 20.981719197707736},
                                                'Logicality of the project description': {'lower': 0.08081020182118388, 'upper': 0.11476803764963517},
                                                'Logicality of the project risks': {'lower': 0.09279613486990718, 'upper': 0.13854156619055974},
                                                'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                                'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Journalism', 'USA'): {'Goal': {'lower': 299.99999999999994, 'upper': 3999.9999999999995},
                                             'Campaign Duration': {'lower': 13.999999999999996, 'upper': 38.99999999999999},
                                             'Length of the project description': {'lower': 343.00000000000017, 'upper': 1226.2999999999997},
                                             'Length of the project risks': {'lower': 87.48800000000003, 'upper': 224.18800000000007},
                                             'Readability score of the project description': {'lower': 10.659999999999998, 'upper': 15.150000000000004},
                                             'Readability score of the project risks': {'lower': 14.591719197707736, 'upper': 21.19271919770773},
                                             'Logicality of the project description': {'lower': 0.086762788184349, 'upper': 0.11712734549247457},
                                             'Logicality of the project risks': {'lower': 0.09351260866160223, 'upper': 0.13815546580445934},
                                             'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 2.546453645613199},
                                             'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Music', 'Others'): {'Goal': {'lower': 371.0000000000005, 'upper': 5000.000000000004},
                                           'Campaign Duration': {'lower': 15.999999999999998, 'upper': 39.99999999999998},
                                           'Length of the project description': {'lower': 299.99999999999994, 'upper': 1138.9999999999998},
                                           'Length of the project risks': {'lower': 74.48799999999999, 'upper': 215.18800000000002},
                                           'Readability score of the project description': {'lower': 9.76, 'upper': 14.053000000000003},
                                           'Readability score of the project risks': {'lower': 13.071719197707736, 'upper': 20.74171919770774},
                                           'Logicality of the project description': {'lower': 0.07796476534105055, 'upper': 0.11994603927353951},
                                           'Logicality of the project risks': {'lower': 0.08555806320705674, 'upper': 0.13265314160821692},
                                           'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                           'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Music', 'USA'): {'Goal': {'lower': 499.99999999999983, 'upper': 6299.999999999996},
                                        'Campaign Duration': {'lower': 17.999999999999996, 'upper': 36.0},
                                        'Length of the project description': {'lower': 291.0000000000001, 'upper': 1003.7999999999997},
                                        'Length of the project risks': {'lower': 76.48800000000003, 'upper': 213.488},
                                        'Readability score of the project description': {'lower': 9.76, 'upper': 13.870000000000001},
                                        'Readability score of the project risks': {'lower': 13.811719197707733, 'upper': 20.92171919770774},
                                        'Logicality of the project description': {'lower': 0.08905961702881461, 'upper': 0.12036539783485084},
                                        'Logicality of the project risks': {'lower': 0.08983613807336693, 'upper': 0.13707321472220824},
                                        'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                        'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Photography', 'Others'): {'Goal': {'lower': 206.00000000000003, 'upper': 3499.9999999999995},
                                                 'Campaign Duration': {'lower': 13.999999999999996, 'upper': 39.99999999999998},
                                                 'Length of the project description': {'lower': 368.5, 'upper': 1608.0000000000002},
                                                 'Length of the project risks': {'lower': 78.98799999999999, 'upper': 206.988},
                                                 'Readability score of the project description': {'lower': 10.207, 'upper': 14.419999999999998},
                                                 'Readability score of the project risks': {'lower': 15.226719197707734, 'upper': 21.68171919770774},
                                                 'Logicality of the project description': {'lower': 0.075832042092638, 'upper': 0.11580648502487545},
                                                 'Logicality of the project risks': {'lower': 0.09072685792809028, 'upper': 0.13990197585919154},
                                                 'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                                 'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Photography', 'USA'): {'Goal': {'lower': 249.9999999999999, 'upper': 3750.0},
                                              'Campaign Duration': {'lower': 10.000000000000002, 'upper': 36.0},
                                              'Length of the project description': {'lower': 349.9999999999999, 'upper': 1203.2999999999997},
                                              'Length of the project risks': {'lower': 80.48799999999997, 'upper': 221.48800000000006},
                                              'Readability score of the project description': {'lower': 10.07, 'upper': 14.240000000000002},
                                              'Readability score of the project risks': {'lower': 16.67171919770774, 'upper': 21.838719197707736},
                                              'Logicality of the project description': {'lower': 0.08352147626958069, 'upper': 0.11726265929097432},
                                              'Logicality of the project risks': {'lower': 0.09658637915340545, 'upper': 0.14212371977271335},
                                              'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                              'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Publishing', 'Others'): {'Goal': {'lower': 299.99999999999994, 'upper': 5000.000000000004},
                                                'Campaign Duration': {'lower': 15.999999999999998, 'upper': 37.99999999999999},
                                                'Length of the project description': {'lower': 365.9999999999999, 'upper': 1509.0000000000005},
                                                'Length of the project risks': {'lower': 75.188, 'upper': 223.78799999999993},
                                                'Readability score of the project description': {'lower': 9.719999999999999, 'upper': 13.989},
                                                'Readability score of the project risks': {'lower': 14.631719197707733, 'upper': 20.911719197707733},
                                                'Logicality of the project description': {'lower': 0.08212956047305887, 'upper': 0.1160730643769324},
                                                'Logicality of the project risks': {'lower': 0.08708737501674237, 'upper': 0.13213620416722016},
                                                'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                                'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Publishing', 'USA'): {'Goal': {'lower': 399.9999999999999, 'upper': 5000.000000000004},
                                             'Campaign Duration': {'lower': 15.0, 'upper': 34.99999999999998},
                                             'Length of the project description': {'lower': 360.99999999999983, 'upper': 1482.4},
                                             'Length of the project risks': {'lower': 76.48800000000003, 'upper': 220.48799999999991},
                                             'Readability score of the project description': {'lower': 9.579999999999998, 'upper': 13.738999999999999},
                                             'Readability score of the project risks': {'lower': 14.601719197707734, 'upper': 20.781719197707737},
                                             'Logicality of the project description': {'lower': 0.08827612065240589, 'upper': 0.11724994016047043},
                                             'Logicality of the project risks': {'lower': 0.09307088185058908, 'upper': 0.13992349975071114},
                                             'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                             'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Technology', 'Others'): {'Goal': {'lower': 299.99999999999994, 'upper': 3499.9999999999995},
                                                'Campaign Duration': {'lower': 19.999999999999996, 'upper': 45.000000000000014},
                                                'Length of the project description': {'lower': 414.99999999999983, 'upper': 1616.5999999999997},
                                                'Length of the project risks': {'lower': 80.78799999999998, 'upper': 263.48799999999994},
                                                'Readability score of the project description': {'lower': 9.395000000000001, 'upper': 12.665000000000001},
                                                'Readability score of the project risks': {'lower': 17.09171919770774, 'upper': 22.071719197707733},
                                                'Logicality of the project description': {'lower': 0.08297649919679558, 'upper': 0.11493780847914228},
                                                'Logicality of the project risks': {'lower': 0.09101260866160223, 'upper': 0.13912071676971036},
                                                'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                                'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Technology', 'USA'): {'Goal': {'lower': 399.9999999999999, 'upper': 3750.0},
                                             'Campaign Duration': {'lower': 21.0, 'upper': 43.99999999999999},
                                             'Length of the project description': {'lower': 397.0000000000001, 'upper': 1564.0000000000005},
                                             'Length of the project risks': {'lower': 82.48799999999999, 'upper': 263.48799999999994},
                                             'Readability score of the project description': {'lower': 9.46, 'upper': 12.799999999999999},
                                             'Readability score of the project risks': {'lower': 17.271719197707736, 'upper': 22.14171919770774},
                                             'Logicality of the project description': {'lower': 0.08780755368637701, 'upper': 0.11560183981054797},
                                             'Logicality of the project risks': {'lower': 0.09147960040953917, 'upper': 0.13845062519052784},
                                             'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                             'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Theater', 'Others'): {'Goal': {'lower': 299.99999999999994, 'upper': 2999.9999999999977},
                                             'Campaign Duration': {'lower': 15.999999999999998, 'upper': 37.99999999999999},
                                             'Length of the project description': {'lower': 302.69999999999993, 'upper': 1214.3000000000002},
                                             'Length of the project risks': {'lower': 83.08800000000001, 'upper': 259.288},
                                             'Readability score of the project description': {'lower': 10.323, 'upper': 14.687},
                                             'Readability score of the project risks': {'lower': 17.38971919770773, 'upper': 22.647719197707733},
                                             'Logicality of the project description': {'lower': 0.08467229776519537, 'upper': 0.11758073196297551},
                                             'Logicality of the project risks': {'lower': 0.09255107020006377, 'upper': 0.13222154053473548},
                                             'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 1.0},
                                             'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}},
                     ('Theater', 'USA'): {'Goal': {'lower': 499.99999999999983, 'upper': 4294.999999999999},
                                          'Campaign Duration': {'lower': 15.0, 'upper': 36.0},
                                          'Length of the project description': {'lower': 321.0, 'upper': 1117.6999999999996},
                                          'Length of the project risks': {'lower': 85.48799999999997, 'upper': 241.48799999999994},
                                          'Readability score of the project description': {'lower': 10.179999999999998, 'upper': 14.91},
                                          'Readability score of the project risks': {'lower': 17.177719197707734, 'upper': 22.61571919770774},
                                          'Logicality of the project description': {'lower': 0.08679217153066147, 'upper': 0.11726378817072722},
                                          'Logicality of the project risks': {'lower': 0.0956950786645287, 'upper': 0.13517927532826882},
                                          'Neutral sentiment of the project description': {'lower': 1.0, 'upper': 2.718281828459045},
                                          'Positive sentiment of the project description': {'lower': 2.718281828459045, 'upper': 2.718281828459045}}}

    # Set the threshold for SHAP weight
    threshold_shap = 0.05

    for i, feature_name in enumerate(final_features):
        if feature_name.startswith("Other"):
            continue

        if feature_name in recommendations_dict and abs(final_values[i]) > threshold_shap:
            # Получаем исходное имя признака для доступа к данным из st.session_state
            original_feature_name = final_original_features[i]

            # Получаем текущее значение пользователя напрямую из st.session_state
            current_value = st.session_state.get(original_feature_name)

            # Для признаков, которые были созданы из категориальных, используем их
            if current_value is None:
                if "Category" in feature_name:
                    # Для категорий значение берется из поля `category`
                    current_value = st.session_state.get('category')
                elif "Country" in feature_name:
                    current_value = st.session_state.get('country')
                elif "sentiment" in feature_name:
                    current_value = st.session_state.get('sentiment_description')

            # Проверяем, что значение существует
            if current_value is None:
                st.write(
                    f"Error in displaying recommendations for {feature_name}.")
                continue

            shap_value = final_values[i]
            unit = units_dict.get(feature_name, "")

            st.subheader(f"{feature_name}")

            # Особая логика для бинарных признаков
            if feature_name in binary_features:
                if shap_value > 0:
                    st.write(f"The positive sentiment of the project description increases the probability of a successful fundraising campaign.")
                else:
                    st.write(f"{recommendations_dict[feature_name]['increase']}")
            else:
                # Логика для непрерывных признаков
                if feature_name in num_features:
                    lower = optimal_range[(st.session_state.category, st.session_state.country)][feature_name]['lower']
                    upper = optimal_range[(st.session_state.category, st.session_state.country)][feature_name]['upper']
                    if lower <= current_value <= upper:
                        if shap_value < 0:
                            st.write(
                                f"The {feature_name} is {current_value:.2f} {unit}. "
                                f"Although this value is within the optimal range from {lower:.2f} to {upper:.2f}, "
                                f"it reduces the probability of project success. "
                                f"We recommend sticking to the middle of the optimal range.")
                        else:
                            st.write(
                                f"The {feature_name} is {current_value:.2f} {unit}. "
                                f"This value falls within the optimal range of {lower:.2f} to {upper:.2f} {unit}, "
                                f"which increases the probability of a successful fundraising campaign.")
                    elif current_value < lower:
                        if shap_value < 0:
                            st.write(
                                f"The {feature_name} is {current_value:.2f} {unit}. "
                                f"This value is below the optimal range of {lower:.2f} to {upper:.2f} {unit}, "
                                f"which decreases the probability of a successful fundraising campaign. "
                                f"\n\n"
                                f"{recommendations_dict[feature_name]['increase']}")
                        else:
                            st.write(
                                f"The {feature_name} is {current_value:.2f} {unit}. "
                                f"While this value increases the probability, it is below the optimal range of {lower:.2f} to {upper:.2f} {unit}. "
                                f"\n\n"
                                f"{recommendations_dict[feature_name]['increase']}")

                    else:  # current_value > upper
                        if shap_value < 0:
                            st.write(
                                f"The {feature_name} is {current_value:.2f} {unit}. "
                                f"This value is above the optimal range of {lower:.2f} to {upper:.2f} {unit}, "
                                f"which decreases the probability of a successful fundraising campaign. "
                                f"\n\n"
                                f"{recommendations_dict[feature_name]['decrease']}")
                        else:
                            st.write(
                                f"The {feature_name} is {current_value:.2f} {unit}. "
                                f"While this value increases the probability, it is above the optimal range of {lower:.2f} to {upper:.2f} {unit}. "
                                f"\n\n"
                                f"{recommendations_dict[feature_name]['decrease']}")

    st.markdown('<div style="margin-top: 56px;"></div>', unsafe_allow_html=True)
    st.button("Edit Data", onclick=navigate_to_input_data)









